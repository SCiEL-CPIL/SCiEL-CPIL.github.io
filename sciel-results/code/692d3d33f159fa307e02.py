# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

_DEADLINE = time.time() + 615.0
_K = 10
_NTH = 70
_NPHI = 130


def _build_harmonics():
    """Return Y (nphi, nth, 2k+1), weights (nth,), and dphi."""
    k = _K
    mu, w = np.polynomial.legendre.leggauss(_NTH)
    theta = np.arccos(mu)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    Y = np.zeros((_NPHI, _NTH, 2 * k + 1), dtype=np.float64)
    for idx, m in enumerate(range(-k, k + 1)):
        am = abs(m)
        N = np.sqrt((2 * k + 1) / (4.0 * np.pi)
                    * float(factorial(k - am, exact=True))
                    / float(factorial(k + am, exact=True)))
        P = lpmv(am, k, mu)
        if m == 0:
            Y[:, :, idx] = N * P
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * N * P * np.cos(m * phi)[:, None]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * N * P * np.sin(am * phi)[:, None]
    return Y, w, dphi


def _objective(c, Y, w, dphi):
    """Compute R(c) = ||f||_4 / ||f||_2."""
    f = Y @ c
    f2 = f * f
    f4 = f2 * f2
    n2 = np.sum(f2 * w, axis=1) * dphi
    n4 = np.sum(f4 * w, axis=1) * dphi
    n2_val = np.sum(n2)
    n4_val = np.sum(n4)
    if n2_val < 1e-12:
        return 1e12
    return (n4_val ** 0.25) / np.sqrt(n2_val)


def _projected_minimize(x0, Y, w, dphi, maxiter=500):
    """Minimize ||f||_4 (scale-invariant proxy for R) on the unit sphere using L-BFGS-B with projection."""
    x0 = x0 / np.linalg.norm(x0)
    best_c = x0.copy()
    best_val = _objective(best_c, Y, w, dphi)
    
    def fun(c):
        # Project to sphere
        nrm = np.linalg.norm(c)
        if nrm < 1e-12:
            return 1e12
        c_proj = c / nrm
        val = _objective(c_proj, Y, w, dphi)
        # Update best if improved
        nonlocal best_c, best_val
        if val < best_val:
            best_val = val
            best_c = c_proj.copy()
        return val

    try:
        # L-BFGS-B is efficient for unconstrained-like problems where we project manually
        # We don't use bounds, just let it move in R^n and project in the function
        res = minimize(fun, x0, method='L-BFGS-B', 
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
        # The 'best_c' tracked inside fun is likely better than res.x if res.x wasn't projected perfectly
        # But res.x should be close. Let's evaluate res.x too just in case.
        res_c = res.x / np.linalg.norm(res.x)
        res_val = _objective(res_c, Y, w, dphi)
        if res_val < best_val:
            best_val = res_val
            best_c = res_c
    except Exception:
        pass
        
    return best_c, best_val


def _make_inits(rng, k):
    """Generate diverse initializations."""
    inits = []
    dim = 2 * k + 1
    
    # 1. Single modes
    for m in range(-k, k + 1):
        v = np.zeros(dim)
        v[m + k] = 1.0
        inits.append(v)
        
    # 2. Pairs of conjugate modes (cos/sin) - these often have flatter magnitude
    for m in range(1, k + 1):
        v = np.zeros(dim)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(v / np.linalg.norm(v))
        
    # 3. m=0 mixed with high m
    for m in [k, k-1, k-2]:
        for weight in [0.1, 0.3, 0.5, 0.7, 0.9]:
            v = np.zeros(dim)
            v[0 + k] = weight
            v[m + k] = np.sqrt(1 - weight**2)
            v[-m + k] = np.sqrt(1 - weight**2)
            inits.append(v / np.linalg.norm(v))
            
    # 4. Random
    for _ in range(40):
        v = rng.standard_normal(dim)
        inits.append(v / np.linalg.norm(v))
        
    # 5. Sparse random
    for _ in range(30):
        v = np.zeros(dim)
        idx = rng.choice(dim, size=rng.integers(2, 8), replace=False)
        v[idx] = rng.standard_normal(len(idx))
        inits.append(v / np.linalg.norm(v))
        
    # 6. Alternating signs
    v = np.ones(dim)
    v[::2] = -1.0
    inits.append(v / np.linalg.norm(v))
    
    # 7. Dominant high m with small noise
    for m in [k, k-1, -(k-1), -k]:
        v = np.zeros(dim)
        v[m + k] = 1.0
        v += 0.1 * rng.standard_normal(dim)
        inits.append(v / np.linalg.norm(v))

    return inits


def rudin_problem10() -> np.ndarray:
    Y, w, dphi = _build_harmonics()
    k = _K
    rng = np.random.default_rng(42)
    inits = _make_inits(rng, k)

    best_c = inits[0].copy()
    best_val = _objective(best_c, Y, w, dphi)
    
    # Phase 1: Optimize from diverse starts
    scored = []
    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 120.0:
            break
        # Normalize start
        x0 = x0 / np.linalg.norm(x0)
        c, val = _projected_minimize(x0, Y, w, dphi, maxiter=300)
        scored.append((val, c))
        if val < best_val:
            best_val = val
            best_c = c
            
    # Sort and keep top candidates for refinement
    scored.sort(key=lambda t: t[0])
    top_candidates = [c for _, c in scored[:10]]
    if not top_candidates:
        top_candidates = [best_c]

    # Phase 2: Refine top candidates with higher precision
    for x0 in top_candidates:
        if time.time() > _DEADLINE - 60.0:
            break
        c, val = _projected_minimize(x0, Y, w, dphi, maxiter=2000)
        if val < best_val:
            best_val = val
            best_c = c
            
    # Phase 3: Random restarts around best
    while time.time() < _DEADLINE - 30.0:
        # Perturb best solution
        noise_scale = 0.3
        x0 = best_c + noise_scale * rng.standard_normal(2 * k + 1)
        x0 = x0 / np.linalg.norm(x0)
        c, val = _projected_minimize(x0, Y, w, dphi, maxiter=500)
        if val < best_val:
            best_val = val
            best_c = c

    out = best_c.astype(np.float64)
    out = out / np.linalg.norm(out)
    
    # Final validity check
    if np.any(np.isnan(out)) or np.any(np.isinf(out)):
        out = np.ones(21) / np.sqrt(21)
    if np.max(np.abs(out)) < 1e-3 or np.max(np.abs(out)) > 1e5:
        out = np.ones(21) / np.sqrt(21)
        
    return out


# EVOLVE-BLOCK-END
