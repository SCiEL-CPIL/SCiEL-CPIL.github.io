# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import eval_hermite

T_START = time.time()
DEADLINE = 210.0 - 15.0

def _now():
    return time.time()

def _time_left():
    return DEADLINE - _now()

def _H(n, x):
    return eval_hermite(n, x)

def _coeff12(c0, c4, c8):
    h0 = _H(0, 0.0)
    h4 = _H(4, 0.0)
    h8 = _H(8, 0.0)
    h12 = _H(12, 0.0)
    if abs(h12) < 1e-15:
        return 0.0
    return -(c0 * h0 + c4 * h4 + c8 * h8) / h12

def _uncertainty_product_analytic(c0, c4, c8):
    """
    Compute uncertainty product using exact Hermite-Gaussian inner products.
    f(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is chosen so f(0) = 0.
    
    Using the non-orthonormal basis, <H_m e^{-x^2/2}, H_n e^{-x^2/2}> = delta_mn * 2^n n! sqrt(pi)
    
    For x^2, we use: x^2 H_n = H_{n+2} + (2n+1) H_n + n(n-1) H_{n-2} (up to scaling)
    More precisely: x^2 phi_n involves phi_{n+2}, phi_n, phi_{n-2}
    
    In the orthonormal basis psi_n = H_n e^{-x^2/2} / sqrt(2^n n! sqrt(pi)):
    <psi_m | x^2 | psi_n> = (n+0.5) delta_mn + 0.5*sqrt(n(n-1)) delta_{m,n-2} + 0.5*sqrt((n+1)(n+2)) delta_{m,n+2}
    
    Since our indices {0,4,8,12} differ by multiples of 4, no off-diagonal terms connect them.
    """
    c12 = _coeff12(c0, c4, c8)
    
    cs = np.array([c0, c4, c8, c12])
    indices = np.array([0, 4, 8, 12])
    
    # Norm factors: 2^n * n!
    norm_factors = (2.0 ** indices) * np.array([math.factorial(n) for n in indices])
    
    # <f, f> = sum c_i^2 * norm_factor_i
    norm_sq = np.sum(cs**2 * norm_factors)
    
    if norm_sq < 1e-15:
        return float('inf')
        
    # <x^2 f, f> = sum c_i^2 * norm_factor_i * (i + 0.5)  [no off-diagonal terms for our index set]
    x2_moment = np.sum(cs**2 * norm_factors * (indices + 0.5))
        
    var_x = x2_moment / norm_sq
    
    # For Fourier-invariant magnitude, Var_p = Var_x, so product = Var_x^2
    return var_x * var_x

def _score(c0, c4, c8):
    try:
        return _uncertainty_product_analytic(c0, c4, c8)
    except Exception:
        return float('inf')

def construct():
    from scipy.optimize import minimize, differential_evolution
    
    def obj(p):
        return _score(p[0], p[1], p[2])
    
    # Try multiple starting points with Nelder-Mead
    candidates = [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (-113.0/100.0, 1.0/25.0, 1.0/3240.0),  # Seed
        (1.0, 0.5, 0.0),
        (-1.0, 0.5, 0.0),
        (1.0, -0.5, 0.0),
        (1.0, 0.0, 0.5),
        (-1.0, 0.0, 0.5),
    ]
    
    best = candidates[0]
    best_val = obj(best)
    
    for c in candidates[1:]:
        v = obj(c)
        if v < best_val:
            best_val = v
            best = c
            
    # Nelder-Mead from best candidate
    if _time_left() > 30.0:
        res = minimize(obj, best, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
        if res.fun < best_val:
            best_val = res.fun
            best = tuple(res.x)
    
    # Try differential evolution if time permits
    if _time_left() > 60.0:
        bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
        de_res = differential_evolution(obj, bounds, seed=42, maxiter=200, 
                                        tol=1e-12, popsize=15, polish=True)
        if de_res.fun < best_val:
            best_val = de_res.fun
            best = tuple(de_res.x)
    
    return np.array(best)
# EVOLVE-BLOCK-END
