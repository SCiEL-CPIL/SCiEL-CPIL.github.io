"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from math import factorial, sqrt, pi

# Pre-compute constants
indices = [0, 4, 8, 12]
M_diag = np.array([sqrt(pi) * (2**n) * factorial(n) for n in indices])
H0_vals = np.array([1.0, -12.0, 2520.0, -1330560.0])

def _compute_bound(z):
    # z are the amplitudes sqrt(c_i^2 M_i) / M_i? 
    # Let's stick to c_i.
    # z_i = c_i * sqrt(M_i)
    # y_i = z_i^2
    # Objective: 2 * (sum i z_i^2 / sum z_i^2)^2
    # Constraint: sum s_i z_i H_i(0) / sqrt(M_i) = 0
    pass

def construct():
    # We optimize over c_0, c_4, c_8. c_12 is determined.
    # We will iterate over sign patterns of c_0, c_4, c_8, c_12.
    # But c_12 is linear in the others.
    # Let's just use differential_evolution on c_0, c_4, c_8 with a good initial guess.
    
    def _objective(c):
        c0, c4, c8 = c
        c12 = -(c0 * 1.0 + c4 * (-12.0) + c8 * 2520.0) / (-1330560.0)
        coeffs = np.array([c0, c4, c8, c12])
        # y_i = c_i^2 M_i
        y = coeffs**2 * M_diag
        sum_y = np.sum(y)
        if sum_y < 1e-12:
            return 1e10
        sum_i_y = np.sum(np.array(indices) * y)
        val = 2.0 * (sum_i_y / sum_y)**2
        return val

    # Use differential evolution
    from scipy.optimize import differential_evolution
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    res = differential_evolution(_objective, bounds, seed=42, maxiter=1000, tol=1e-10, polish=True)
    
    best_x = res.x
    best_val = res.fun
    
    # Refine with Nelder-Mead
    res2 = minimize(_objective, best_x, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
    if res2.fun < best_val:
        best_x = res2.x
        best_val = res2.fun
        
    return best_x

# EVOLVE-BLOCK-END
