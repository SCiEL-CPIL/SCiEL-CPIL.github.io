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


def _f_on_grid(Y, w, dphi, c):
    """Compute f values and weighted sums needed for ratio and gradient."""
    f = Y @ c
    f2 = f * f
    f3 = f2 * f
    f4 = f2 * f2
    w_ext = w[None, :]
    N2 = np.sum(f2 * w_ext) * dphi
    N4 = np.sum(f4 * w_ext) * dphi
    f3Y = np.sum(f3[:, :, None] * Y * w_ext[:, :, None], axis=(0, 1)) * dphi
    fY = np.sum(f[:, :, None] * Y * w_ext[:, :, None], axis=(0, 1)) * dphi
    return f, N2, N4, f3Y, fY


def _ratio_val_and_grad(Y, w, dphi, c):
    """Compute R = ||f||4/||f||2 and its projected gradient on the sphere."""
    f, N2, N4, f3Y, fY = _f_on_grid(Y, w, dphi, c)
    if N2 < 1e-15:
        return np.inf, np.zeros_like(c)
    R = np.sqrt(N4 / N2)
    grad_R2 = 4.0 * f3Y / N2 - 2.0 * N4 * fY / (N2 * N2)
    gc = np.dot(grad_R2, c)
    grad_tan = grad_R2 - gc * c
    return R, grad_tan


def _trust_region_sphere(c, Y, w, dphi, delta0=0.5, max_iter=500, tol=1e-14):
    """Trust-region method on the unit sphere for minimizing R(c)."""
    c = c / np.linalg.norm(c)
    best_c = c.copy()
    best_val = _ratio_val_and_grad(Y, w, dphi, c)[0]
    delta = delta0

    for it in range(max_iter):
        if time.time() > _DEADLINE - 30.0:
            break
        val, g = _ratio_val_and_grad(Y, w, dphi, c)
        if val < best_val:
            best_val = val
            best_c = c.copy()
        
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        d = -g / g_norm
        t_max = min(delta, np.pi * 0.99)
        
        t_vals = np.array([0.0, t_max * 0.2, t_max * 0.4, t_max * 0.6, t_max * 0.8, t_max])
        t_vals = np.clip(t_vals, 0, np.pi * 0.999)
        f_vals = np.array([_ratio_val_and_grad(Y, w, dphi,
                     (np.cos(t) * c + np.sin(t) * d) / np.linalg.norm(np.cos(t) * c + np.sin(t) * d))[0]
                     for t in t_vals])
        
        best_idx = np.argmin(f_vals)
        t_best = t_vals[best_idx]
        
        if 0 < best_idx < len(t_vals) - 1:
            t_lo = t_vals[best_idx - 1]
            t_hi = t_vals[best_idx + 1]
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            for _ in range(25):
                if time.time() > _DEADLINE - 30.0:
                    break
                t1 = t_hi - gr * (t_hi - t_lo)
                t2 = t_lo + gr * (t_hi - t_lo)
                c1 = (np.cos(t1) * c + np.sin(t1) * d) / np.linalg.norm(np.cos(t1) * c + np.sin(t1) * d)
                c2 = (np.cos(t2) * c + np.sin(t2) * d) / np.linalg.norm(np.cos(t2) * c + np.sin(t2) * d)
                f1 = _ratio_val_and_grad(Y, w, dphi, c1)[0]
                f2 = _ratio_val_and_grad(Y, w, dphi, c2)[0]
                if f1 < f2:
                    t_hi = t2
                else:
                    t_lo = t1
            t_best = (t_lo + t_hi) / 2.0
        
        c_new = (np.cos(t_best) * c + np.sin(t_best) * d) / np.linalg.norm(np.cos(t_best) * c + np.sin(t_best) * d)
        val_new = _ratio_val_and_grad(Y, w, dphi, c_new)[0]
        
        if val_new < best_val:
            best_val = val_new
            best_c = c_new.copy()
        
        if t_best < 0.1 * delta:
            delta = max(delta * 0.5, 1e-10)
        elif t_best > 0.9 * delta:
            delta = min(delta * 1.5, 1.0)
        
        c = c_new
    
    return best_c, best_val


def _make_inits(rng, k):
    """Diverse initial directions with emphasis on highest-order harmonics."""
    inits = []
    for m in [k, -k]:
        v = np.zeros(2 * k + 1)
        v[m + k] = 1.0
        inits.append(v)
    v = np.zeros(2 * k + 1)
    v[k + k] = 1.0
    v[-k + k] = 1.0
    inits.append(v / np.linalg.norm(v))
    for m in [k-1, -(k-1)]:
        v = np.zeros(2 * k + 1)
        v[m + k] = 1.0
        inits.append(v)
    v = np.zeros(2 * k + 1)
    v[k-1 + k] = 1.0
    v[-(k-1) + k] = 1.0
    inits.append(v / np.linalg.norm(v))
    for m in [k-2, -(k-2)]:
        v = np.zeros(2 * k + 1)
        v[m + k] = 1.0
        inits.append(v)
    v = np.zeros(2 * k + 1)
    v[k-2 + k] = 1.0
    v[-(k-2) + k] = 1.0
    inits.append(v / np.linalg.norm(v))
    for m in [k, k-1, k-2]:
        for weight in [0.1, 0.5, 0.9]:
            v = np.zeros(2 * k + 1)
            v[0 + k] = weight
            v[m + k] = np.sqrt(1 - weight**2)
            v[-m + k] = np.sqrt(1 - weight**2)
            inits.append(v / np.linalg.norm(v))
    for _ in range(20):
        v = rng.standard_normal(2 * k + 1)
        inits.append(v / np.linalg.norm(v))
    for m in range(-k + 3, k - 2):
        v = np.zeros(2 * k + 1)
        v[m + k] = 1.0
        inits.append(v)
    for m in range(1, k + 1):
        v = np.zeros(2 * k + 1)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(v / np.linalg.norm(v))
    v = np.ones(2 * k + 1)
    v[::2] = -1.0
    inits.append(v / np.linalg.norm(v))
    for _ in range(8):
        v = np.zeros(2 * k + 1)
        idx = rng.choice(2 * k + 1, size=rng.integers(2, 8), replace=False)
        v[idx] = rng.standard_normal(len(idx))
        if np.linalg.norm(v) > 0:
            inits.append(v / np.linalg.norm(v))
    return inits


def rudin_problem10() -> np.ndarray:
    Y, w, dphi = _build_harmonics()
    k = _K
    rng = np.random.default_rng(12345)
    inits = _make_inits(rng, k)

    best_c = inits[0].copy()
    best_val = np.inf

    # Phase 1: trust-region from many diverse starts
    scored = []
    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 200.0:
            break
        c, val = _trust_region_sphere(x0, Y, w, dphi, delta0=0.5, max_iter=200)
        scored.append((val, c))
    scored.sort(key=lambda t: t[0])
    top = [c for _, c in scored[:8]]
    if not top:
        top = [best_c]

    # Phase 2: refine top candidates
    for x0 in top:
        if time.time() > _DEADLINE - 120.0:
            break
        c, val = _trust_region_sphere(x0, Y, w, dphi, delta0=0.3, max_iter=500, tol=1e-15)
        if val < best_val:
            best_val = val
            best_c = c

    # Phase 3: random restarts
    while time.time() < _DEADLINE - 60.0:
        x0 = best_c + 0.5 * rng.standard_normal(2 * k + 1)
        x0 = x0 / np.linalg.norm(x0)
        c, val = _trust_region_sphere(x0, Y, w, dphi, delta0=0.5, max_iter=300)
        if val < best_val:
            best_val = val
            best_c = c

    # Phase 4: High-precision polish
    if time.time() < _DEADLINE - 15.0:
        c, val = _trust_region_sphere(best_c, Y, w, dphi, delta0=0.1, max_iter=1000, tol=1e-16)
        if val < best_val:
            best_val = val
            best_c = c

    out = best_c.astype(np.float64)
    out = out / np.linalg.norm(out)
    return out


# EVOLVE-BLOCK-END
