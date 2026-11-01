import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 195.0  # Leave 15s safety margin for 210s total

def construct():
    """
    Finds coefficients c0, c4, c8 for the Hermite combination
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    such that g(0)=0 and the upper bound on A(g)*A(hat{g}) is minimized.
    Returns np.array([c0, c4, c8]).
    """
    
    # Precompute Gauss-Hermite nodes and weights for fast integration
    # Integral is approx 2*sqrt(2) * sum(w_i * u_i^4 * |P(u_i*sqrt(2))|)
    N = 400
    u, w = np.polynomial.hermite.hermgauss(N)
    sqrt2 = np.sqrt(2.0)
    u_scaled = u * sqrt2
    u4 = u**4
    
    def get_c12(c0, c4, c8):
        return -(c0 + 3.0*c4 + 105.0*c8) / 945.0

    def compute_p_vec(x, c0, c4, c8, c12):
        """Vectorized computation of P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12"""
        x2 = x * x
        x4 = x2 * x2
        x6 = x4 * x2
        x8 = x4 * x4
        x10 = x8 * x2
        x12 = x6 * x6
        
        h0 = np.ones_like(x)
        h4 = x4 - 6.0 * x2 + 3.0
        h8 = x8 - 28.0 * x6 + 210.0 * x4 - 420.0 * x2 + 105.0
        h12 = x12 - 84.0 * x10 + 2520.0 * x8 - 30240.0 * x6 + 151200.0 * x4 - 221760.0 * x2 + 10395.0
        
        return c0 * h0 + c4 * h4 + c8 * h8 + c12 * h12

    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        p_vals = compute_p_vec(u_scaled, c0, c4, c8, c12)
        abs_p = np.abs(p_vals)
        
        # 2*sqrt(2) is the scaling factor derived from change of variables
        integral_val = 2.0 * sqrt2 * np.sum(w * u4 * abs_p)
        
        return 2.0 * integral_val

    best_score = np.inf
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    seed_score = objective(best_coeffs)
    if seed_score < best_score:
        best_score = seed_score

    np.random.seed(42)
    
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    # Phase 1: Multi-start Nelder-Mead with faster objective
    num_starts = 100
    
    for i in range(num_starts):
        if time.time() > TIME_LIMIT - 10:
            break
            
        x0 = np.array([
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1]),
            np.random.uniform(bounds[2][0], bounds[2][1])
        ])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Phase 2: L-BFGS-B polish
    if time.time() < TIME_LIMIT - 5:
        try:
            res2 = minimize(objective, best_coeffs, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res2.fun < best_score:
                best_score = res2.fun
                best_coeffs = res2.x.copy()
        except Exception:
            pass

    return best_coeffs
