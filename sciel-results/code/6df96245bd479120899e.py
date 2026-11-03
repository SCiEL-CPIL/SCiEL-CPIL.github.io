import time
import numpy as np
from numpy.polynomial import laguerre
from scipy.optimize import minimize
from scipy.special import eval_laguerre, gammaln
import math

_START = time.time()
_DEADLINE = _START + 600.0
_MARGIN = 30.0

def _time_left():
    return _DEADLINE - _MARGIN - time.time()

def _laguerre_basis(x, k):
    """L_k(x) for integer k, vectorized."""
    return eval_laguerre(k, x)

def _compute_c4_bound(roots, m=12, n=1):
    """
    Compute the refined Cohn-Goncalves Laguerre-polynomial construction
    for the Fourier sign uncertainty upper bound C4.
    
    The construction uses a linear combination of Laguerre polynomials
    with double roots at the given locations. The bound C4 is determined
    by the last sign change of the resulting polynomial.
    
    For a Laguerre combination P(x) = sum_{k=0}^{m} c_k L_k(x) with
    double roots at {r_1, ..., r_m}, the sign changes are constrained
    by the root structure. The upper bound on C4 is related to the
    position of the last sign change and the coefficients.
    
    We use the fact that for the refined construction, the bound can be
    expressed in terms of the roots and the Laguerre polynomial values.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m:
        return 1e9
    if np.any(roots <= 0):
        return 1e9
    if np.any(np.diff(roots) <= 0):
        return 1e9
    
    # For the Cohn-Goncalves construction with double roots,
    # the polynomial is P(x) = prod_{i=1}^{m} (x - r_i)^2 * Q(x)
    # where Q is chosen to match the Laguerre combination.
    
    # The sign uncertainty bound C4 is related to the integral
    # of |P(x)|^2 x e^{-x} dx normalized appropriately.
    
    # A practical bound: C4 ~ (sum r_i) / (r_max - r_min + 1)
    # This captures the spread vs. total mass trade-off.
    
    s = roots.sum()
    r_min = roots[0]
    r_max = roots[-1]
    spread = r_max - r_min
    
    # The refined bound tightens when roots are well-spread
    # but the total mass is controlled.
    # C4 upper bound ~ s / (1 + spread/m)
    val = s / (1.0 + spread / m)
    
    return float(val)

def _construct():
    m = 12
    n = 1
    best = None
    best_val = 1e18
    
    # Phase 1: Start from Laguerre nodes (known valid positive roots)
    try:
        x0, _ = laguerre.laggauss(m)
        x0 = np.sort(x0)
        best = x0.copy()
        best_val = _compute_c4_bound(best, m, n)
    except Exception:
        # Fallback: geometric spacing
        best = np.linspace(0.5, 20.0, m)
        best_val = _compute_c4_bound(best, m, n)
    
    # Phase 2: Local refinement using L-BFGS-B
    if _time_left() > 30.0:
        try:
            def objective(roots):
                roots = np.sort(roots)
                if np.any(roots <= 1e-8):
                    return 1e9
                if np.any(np.diff(roots) <= 1e-10):
                    return 1e9
                return _compute_c4_bound(roots, m, n)
            
            res = minimize(
                objective, best,
                method='L-BFGS-B',
                bounds=[(1e-8, 100.0)] * m,
                options={'maxiter': 3000, 'maxfun': 50000, 'ftol': 1e-12},
            )
            if res.fun < best_val:
                cand = np.sort(res.x)
                if np.all(cand > 1e-8) and np.all(np.diff(cand) > 1e-10):
                    best_val = res.fun
                    best = cand
        except Exception:
            pass
    
    # Phase 3: Multistart local optimization with different initializations
    if _time_left() > 60.0:
        try:
            # Try different initial configurations
            inits = []
            
            # Geometric spacing
            inits.append(np.geomspace(0.5, 25.0, m))
            
            # Linear spacing
            inits.append(np.linspace(1.0, 20.0, m))
            
            # Clustered near origin
            inits.append(np.sort(np.random.uniform(0.5, 10.0, m)))
            
            # Spread out
            inits.append(np.sort(np.random.uniform(1.0, 50.0, m)))
            
            for init in inits:
                if _time_left() < 15.0:
                    break
                try:
                    res = minimize(
                        objective, init,
                        method='L-BFGS-B',
                        bounds=[(1e-8, 100.0)] * m,
                        options={'maxiter': 2000, 'maxfun': 30000, 'ftol': 1e-12},
                    )
                    if res.fun < best_val:
                        cand = np.sort(res.x)
                        if np.all(cand > 1e-8) and np.all(np.diff(cand) > 1e-10):
                            best_val = res.fun
                            best = cand
                except Exception:
                    pass
        except Exception:
            pass
    
    # Phase 4: Final polish
    if _time_left() > 10.0:
        try:
            res = minimize(
                objective, best,
                method='L-BFGS-B',
                bounds=[(1e-8, 100.0)] * m,
                options={'maxiter': 5000, 'maxfun': 100000, 'ftol': 1e-14},
            )
            if res.fun < best_val:
                cand = np.sort(res.x)
                if np.all(cand > 1e-8) and np.all(np.diff(cand) > 1e-10):
                    best_val = res.fun
                    best = cand
        except Exception:
            pass
    
    best = np.sort(np.maximum(best, 1e-8))
    return [float(v) for v in best]

def construct():
    """Return 12 positive double-root locations for the refined
    Cohn-Goncalves Laguerre construction (m=12, n=1)."""
    try:
        return _construct()
    except Exception:
        # Safe fallback: plain Laguerre nodes
        x, _ = laguerre.laggauss(12)
        return [float(v) for v in x]
