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


def _slsqp_minimize(x0, Y, w, dphi, maxiter=1000):
    """Minimize R(c) subject to ||c||^2 = 1 using SLSQP."""
    
    def fun(c):
        return _objective(c, Y, w, dphi)

    def cons(c):
        return np.sum(c * c) - 1.0

    constraints = ({'type': 'eq', 'fun': cons})
    
    try:
        res = minimize(fun, x0, method='SLSQP', constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-12})
        if res.success:
            return res.x, res.fun
        else:
            # If SLSQP fails, return the best point found or original
            return x0, _objective(x0, Y, w, dphi)
    except Exception:
        return x0, _objective(x0, Y, w, dphi)


def _make_inits(rng, k):
    """Generate diverse initializations."""
    inits = []
    dim = 2 * k + 1
    
    # 1. Single modes
    for m in range(-k, k + 1):
        v = np.zeros(dim)
        v[m + k] = 1.0
        inits.append(v)
        
    # 2. Pairs of conjugate modes (cos/sin)
    for m in range(1, k + 1):
        v = np.zeros(dim)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(v / np.linalg.norm(v))
        
    # 3. m=0 mixed with high m
    for m in [k, k-1, k-2]:
        for weight in [0.1, 0.5, 0.9]:
            v = np.zeros(dim)
            v[0 + k] = weight
            v[m + k] = np.sqrt(1 - weight**2)
            v[-m + k] = np.sqrt(1 - weight**2)
            inits.append(v / np.linalg.norm(v))
            
    # 4. Random
    for _ in range(30):
        v = rng.standard_normal(dim)
        inits.append(v / np.linalg.norm(v))
        
    # 5. Sparse random
    for _ in range(20):
        v = np.zeros(dim)
        idx = rng.choice(dim, size=rng.integers(2, 6), replace=False)
        v[idx] = rng.standard_normal(len(idx))
        inits.append(v / np.linalg.norm(v))
        
    # 6. Alternating signs
    v = np.ones(dim)
    v[::2] = -1.0
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
    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 120.0:
            break
        # Normalize start
        x0 = x0 / np.linalg.norm(x0)
        c, val = _slsqp_minimize(x0, Y, w, dphi, maxiter=500)
        if val < best_val:
            best_val = val
            best_c = c
            
    # Phase 2: Refine best with higher precision
    if time.time() < _DEADLINE - 60.0:
        c, val = _slsqp_minimize(best_c, Y, w, dphi, maxiter=2000)
        if val < best_val:
            best_val = val
            best_c = c
            
    # Phase 3: Random restarts around best
    while time.time() < _DEADLINE - 30.0:
        x0 = best_c + 0.3 * rng.standard_normal(2 * k + 1)
        x0 = x0 / np.linalg.norm(x0)
        c, val = _slsqp_minimize(x0, Y, w, dphi, maxiter=200)
        if val < best_val:
            best_val = val
            best_c = c

    out = best_c.astype(np.float64)
    out = out / np.linalg.norm(out)
    return out


# EVOLVE-BLOCK-END
