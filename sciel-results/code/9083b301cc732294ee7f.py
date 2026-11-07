"""Optimized uncertainty inequality: 2D ratio search with y=x² substitution."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

# Physicists' Hermite polynomials in y = x^2 (descending powers y^6 .. y^0)
# H_0(y) = 1
# H_4(y) = 16y^2 - 48y + 12
# H_8(y) = 256y^4 - 1344y^3 + 2688y^2 - 1344y + 112
# H_12(y) = 4096y^6 - 147456y^5 + 2257920y^4 - 15930720y^3 + 47056320y^2 - 54432000y + 10321920

_H12 = np.array([4096.0, -147456.0, 2257920.0, -15930720.0, 47056320.0, -54432000.0, 10321920.0])
_H8 = np.array([0.0, 0.0, 256.0, -1344.0, 2688.0, -1344.0, 112.0])
_H4 = np.array([0.0, 0.0, 0.0, 0.0, 16.0, -48.0, 12.0])
_H0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

_CONST = 10321920.0  # H_12(0)


def _build_P(p, q):
    """Build P(y) coefficients (descending y^6..y^0) for g(x)/x^2.
    c0=1, c1=p, c2=q, c3 fixed by g(0)=0."""
    c3 = -(1.0 + 12.0 * p + 112.0 * q) / _CONST
    return _H12 * c3 + _H8 * q + _H4 * p + _H0


def _largest_positive_sign_root(c):
    """Largest positive root of P(y) where P changes sign (odd multiplicity).
    Returns penalty if none found."""
    # Trim negligible leading coefficients
    max_abs = np.max(np.abs(c))
    if max_abs < 1e-30:
        return 1e18
    nz = np.where(np.abs(c) > 1e-12 * max_abs)[0]
    if len(nz) == 0:
        return 1e18
    c_trimmed = c[nz[0]:]
    
    try:
        rts = np.roots(c_trimmed)
    except Exception:
        return 1e18
    
    best = -1.0
    for rt in rts:
        if abs(rt.imag) > 1e-9 * max(1.0, abs(rt.real)):
            continue
        r = rt.real
        if r <= 1e-12:
            continue
        d = 1e-7 * max(1.0, abs(r))
        try:
            a = np.polyval(c_trimmed, r - d)
            b = np.polyval(c_trimmed, r + d)
        except Exception:
            continue
        if a * b < 0.0 and r > best:
            best = r
    
    return best if best > 0.0 else 1e18


def _objective(x):
    p, q = x
    P = _build_P(p, q)
    return _largest_positive_sign_root(P)


def _seed_ratios():
    c0, c1, c2 = -113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0
    return np.array([c1 / c0, c2 / c0])


def construct():
    t0 = time.time()
    deadline = 170.0  # safety margin under 210s kill
    
    seed = _seed_ratios()
    best_x = seed.copy()
    best_val = _objective(best_x)
    
    # Phase 1: Local Nelder-Mead from seed
    if time.time() - t0 < deadline - 30:
        try:
            res = minimize(_objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Random restarts with local optimization
    rng = np.random.RandomState(42)
    for i in range(15):
        if time.time() - t0 > deadline - 30:
            break
        # Varying scales for exploration
        scale = 0.5 / (1.0 + 0.3 * i)
        x0 = best_x + rng.randn(2) * scale
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Differential evolution for broader search
    if time.time() - t0 < deadline - 60:
        try:
            bounds = [(-5.0, 5.0), (-0.2, 0.2)]
            de_res = differential_evolution(_objective, bounds, seed=42,
                                             maxiter=80, popsize=30,
                                             tol=1e-14, polish=True,
                                             mutation=(0.5, 1.0),
                                             recombination=0.7)
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_x = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Broader differential evolution
    if time.time() - t0 < deadline - 40:
        try:
            bounds = [(-10.0, 10.0), (-0.5, 0.5)]
            de_res = differential_evolution(_objective, bounds, seed=123,
                                             maxiter=60, popsize=25,
                                             tol=1e-13, polish=True,
                                             mutation=(0.8, 1.2))
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_x = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish
    if time.time() - t0 < deadline - 5:
        try:
            res = minimize(_objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-18,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Convert back to c0, c1, c2
    p, q = best_x
    c0 = 1.0
    c1 = p
    c2 = q
    
    return np.array([c0, c1, c2])
# EVOLVE-BLOCK-END
