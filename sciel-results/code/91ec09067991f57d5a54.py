"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import differential_evolution, minimize
import time

# Precompute Gauss-Hermite nodes and weights
# N=20 is sufficient for high accuracy (integrates up to degree 39)
N_QUAD = 20
_x_nodes, _w_weights = hermgauss(N_QUAD)

# Precompute Hermite polynomial values at nodes
_H0 = np.ones(N_QUAD)
_H4 = eval_hermite(4, _x_nodes)
_H8 = eval_hermite(8, _x_nodes)
_H12 = eval_hermite(12, _x_nodes)

# Values at x=0 for constraint (Physicist's Hermite)
_h0_0 = 1.0
_h4_0 = 3.0
_h8_0 = 105.0
_h12_0 = 945.0

# Precompute P^2 terms and x^2 * P^2 terms structure for speed?
# P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
# c12 = -(c0 + 3*c4 + 105*c8) / 945
# P = c0*(H0 - H12/945) + c4*(H4 - 3*H12/945) + c8*(H8 - 105*H12/945)
# Let A = H0 - H12/945
# Let B = H4 - 3*H12/945
# Let C = H8 - 105*H12/945
# Then P = c0*A + c4*B + c8*C
# This linear combination is faster to compute.

_A = _H0 - _H12 / _h12_0
_B = _H4 - _h4_0 * _H12 / _h12_0
_C = _H8 - _h8_0 * _H12 / _h12_0

_x_sq = _x_nodes**2

def _compute_bound(c0, c4, c8):
    """Compute the upper bound C4 for given coefficients."""
    # Compute P(x) at nodes
    P = c0 * _A + c4 * _B + c8 * _C
    
    P_sq = P * P
    norm_sq = np.sum(_w_weights * P_sq)
    
    if norm_sq < 1e-15:
        return 1e10
        
    second_moment = np.sum(_w_weights * _x_sq * P_sq) / norm_sq
    C4 = second_moment ** 2
    return C4

def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 210s kill
    
    # Define bounds for Differential Evolution
    # The coefficients can be of any scale, but the objective is scale-invariant.
    # We can constrain the vector norm or just use wide bounds.
    # Using wide bounds allows DE to explore the direction space effectively.
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    best_c = None
    best_val = float('inf')
    
    # 1. Global Optimization with Differential Evolution
    # This is robust against local minima.
    try:
        # maxiter=1000, popsize=15 is a good balance for 3D problems
        res = differential_evolution(
            _compute_bound, 
            bounds, 
            maxiter=1000, 
            popsize=20, 
            tol=1e-10, 
            seed=42,
            polish=False, # We will polish manually to save time and control it
            init='sobol' # Better sampling than 'latinhypercube' or 'sobol' (actually 'sobol' is better)
        )
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
    except Exception:
        pass
        
    # Check time
    if time.time() > deadline - 10:
        if best_c is not None:
            return np.array(best_c)
        # Fallback to seed if DE failed
        return np.array([-1.13, 0.04, 0.00030864])

    # 2. Local Polish with Nelder-Mead or L-BFGS-B
    # DE gives a good starting point. Local refinement ensures high precision.
    if best_c is not None:
        # Try Nelder-Mead for final polish
        try:
            res_local = minimize(
                _compute_bound, 
                best_c, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res_local.fun < best_val:
                best_val = res_local.fun
                best_c = res_local.x.copy()
        except Exception:
            pass
            
        # Try L-BFGS-B as well, it might find a slightly better point if NM got stuck
        try:
            res_lbgfs = minimize(
                _compute_bound, 
                best_c, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res_lbgfs.fun < best_val:
                best_val = res_lbgfs.fun
                best_c = res_lbgfs.x.copy()
        except Exception:
            pass
            
    # If best_c is still None (unlikely), return seed
    if best_c is None:
        return np.array([-1.13, 0.04, 0.00030864])
        
    return np.array(best_c)

# EVOLVE-BLOCK-END
