"""Analytic Hermite-Gaussian uncertainty bound with 2D global optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

_DEADLINE = time.time() + 200

def _M0_M2(u, v):
    """
    Computes M0 and M2 for a0=1, a4=u, a8=v.
    a12 = -(1 + 3u + 105v)/945
    """
    a12 = -(1.0 + 3.0 * u + 105.0 * v) / 945.0
    
    M0 = 1.0 + 3.0 * u * u + 105.0 * v * v + 945.0 * a12 * a12
    
    M2 = 3.0 + 24.0 * u + 105.0 * u * u + 1680.0 * u * v + 2940.0 * v * v + 11340.0 * v * a12
    
    return M0, M2

def _objective_2d(params):
    u, v = params
    M0, M2 = _M0_M2(u, v)
    if M0 <= 0:
        return 1e18
    return M2 / M0

def construct():
    # We minimize vf = M2/M0. The total bound J = sqrt(vf * 7/3).
    # Minimizing vf minimizes J.
    # Bounds for u, v. The function is smooth and bounded.
    # Typical values are small.
    bounds = [(-10.0, 10.0), (-10.0, 10.0)]
    
    best_val = 1e18
    best_params = np.array([0.0, 0.0])
    
    # Use differential evolution for global search
    # maxiter is number of generations
    # popsize is population size
    try:
        res = differential_evolution(_objective_2d, bounds, maxiter=1000, popsize=20, tol=1e-12, seed=42, polish=True)
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x.copy()
    except Exception:
        pass

    # Convert back to a0, a4, a8
    # We fixed a0=1.
    # The returned coefficients should be (1, u, v).
    # However, the original interface expects (a0, a4, a8).
    # Since the function is homogeneous, any scalar multiple is equivalent.
    # We return (1, u, v).
    
    # Check deadline
    if time.time() > _DEADLINE - 10:
         return np.array([1.0, best_params[0], best_params[1]])
         
    # Optional: Polish with local optimizer on the 2D problem
    # The 'polish=True' in differential_evolution does this.
    
    return np.array([1.0, best_params[0], best_params[1]])
# EVOLVE-BLOCK-END
