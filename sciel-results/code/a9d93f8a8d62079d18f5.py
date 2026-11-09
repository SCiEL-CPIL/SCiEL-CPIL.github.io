"""Refined Cohn-Goncalves Laguerre-polynomial construction."""
import time
import numpy as np
from scipy.special import roots_genlaguerre
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_HARD_DEADLINE = 620.0  # safety margin before 630s kill
M = 12
NROOTS = 12

def _deadline_reached():
    return (time.time() - _START_TIME) >= _HARD_DEADLINE

def _compute_bound(r):
    """
    Compute the C4 moment ratio for given double-root locations.
    
    For the Cohn-Goncalves construction with m=12, n=1, we have a polynomial
    P(x) = prod_{i=1}^{12} (x - r_i)^2.
    
    The uncertainty bound C4 involves the ratio:
    C4 = [integral_0^inf x^4 e^{-x} |P(x)|^2 dx] / [integral_0^inf e^{-x} |P(x)|^2 dx]
    
    We use Gauss-Laguerre quadrature for integrals with e^{-x} weight.
    Since P has degree 24, we need sufficient quadrature points.
    """
    r = np.asarray(r, dtype=float)
    
    # Use Gauss-Laguerre quadrature with enough points to resolve the polynomial
    # P has degree 24, so |P|^2 has degree 48. We need at least 25 points for
    # exact integration of degree 48 polynomials, but use more for safety.
    n_quad = 80
    x_nodes, x_weights = np.polynomial.laguerre.laggauss(n_quad)
    
    # Evaluate P(x) at each node
    P_vals = np.ones_like(x_nodes)
    for ri in r:
        P_vals *= (x_nodes - ri) ** 2
    
    # Compute integrals
    P2 = P_vals ** 2
    
    num = np.sum(x_weights * x_nodes**4 * P2)
    den = np.sum(x_weights * P2)
    
    if den < 1e-30:
        return 1e10  # very large penalty
    
    return num / den

def _objective(r_flat):
    """
    Objective function for optimization.
    Enforces positivity and ordering of roots.
    """
    if _deadline_reached():
        return 1e10
    
    r = np.asarray(r_flat, dtype=float)
    
    # Enforce positivity
    r = np.maximum(r, 1e-6)
    
    # Enforce strictly increasing order
    r = np.maximum.accumulate(r)
    # Ensure strict increase by adding small epsilon
    for i in range(1, len(r)):
        if r[i] <= r[i-1]:
            r[i] = r[i-1] + 1e-6
    
    return _compute_bound(r)

def construct():
    """
    Returns a list of 12 positive double-root locations for the Laguerre combination.
    Uses the roots of the generalized Laguerre polynomial L_12^(1)(x) as initial
    positions, then refines them using a constrained optimizer to minimize the
    C4 moment ratio computed via high-precision Gauss-Laguerre quadrature.
    """
    m = 12
    alpha = 1.0
    
    try:
        # Compute roots of L_m^(alpha)(x)
        x, w = roots_genlaguerre(m, alpha)
        roots = sorted(x)
        
        # Verify constraints: 12 roots, all positive
        if len(roots) != 12:
            raise ValueError("Expected 12 roots")
        if any(r <= 0 for r in roots):
            raise ValueError("Roots must be positive")
            
        best_r = np.array(roots, dtype=float)
        best_val = _compute_bound(best_r)
        
        # Try optimization to refine the root positions
        if not _deadline_reached():
            # Use Nelder-Mead for derivative-free optimization
            try:
                res = minimize(
                    _objective,
                    best_r,
                    method="Nelder-Mead",
                    options=dict(
                        maxiter=2000,
                        xatol=1e-8,
                        fatol=1e-12,
                        adaptive=True
                    )
                )
                if res is not None and np.all(np.isfinite(res.x)):
                    val = _compute_bound(res.x)
                    if val < best_val:
                        best_val = val
                        best_r = res.x.copy()
            except Exception:
                pass
        
        # Final polish with Powell's method if time allows
        if not _deadline_reached() and (time.time() - _START_TIME) < _HARD_DEADLINE - 30:
            try:
                res = minimize(
                    _objective,
                    best_r,
                    method="Powell",
                    options=dict(
                        maxiter=5000,
                        ftol=1e-14,
                        maxfev=100000
                    )
                )
                if res is not None and np.all(np.isfinite(res.x)):
                    val = _compute_bound(res.x)
                    if val < best_val:
                        best_val = val
                        best_r = res.x.copy()
            except Exception:
                pass
        
        # Ensure final roots are valid
        best_r = np.maximum(best_r, 1e-6)
        best_r = np.maximum.accumulate(best_r)
        for i in range(1, len(best_r)):
            if best_r[i] <= best_r[i-1]:
                best_r[i] = best_r[i-1] + 1e-6
        
        return [float(r) for r in best_r]
        
    except Exception:
        # Fallback to seed program behavior if something goes wrong
        return [4.0 * (i + 1) for i in range(NROOTS)]
# EVOLVE-BLOCK-END
