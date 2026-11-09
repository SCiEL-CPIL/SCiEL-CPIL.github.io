import numpy as np
from scipy.special import roots_genlaguerre, eval_genlaguerre
from scipy.optimize import minimize
import time

def construct():
    t0 = time.time()
    DEADLINE = 590
    
    m = 12
    alpha = 1
    
    # Fallback: Seed program
    seed_roots = np.array([4.0 * (i + 1) for i in range(m)])
    
    try:
        # Structural Change: Use roots of the Laguerre polynomial L_m^(alpha)(x)
        # These are the zeros of the orthogonal polynomial, which often
        # minimize the energy or norm of the associated expansion.
        roots = roots_genlaguerre(m, alpha)
        roots = np.sort(roots)
        
        # Sanity check
        if np.any(roots <= 0):
            return list(seed_roots)
            
        # Define a cost function to potentially refine the roots.
        # We assume the "upper bound on C4" is correlated with the magnitude of
        # the Laguerre expansion coefficients of P(x) = prod (x-r_i)^2.
        # We minimize the L1 norm of the coefficients.
        
        def get_coeffs(r):
            poly = np.array([1.0])
            for ri in r:
                factor = np.array([1.0, -2*ri, ri*ri])
                poly = np.convolve(poly, factor)
            
            deg = 2 * m
            x_eval = np.linspace(0.1, 50.0, deg + 1)
            L_matrix = np.zeros((deg + 1, deg + 1))
            for k in range(deg + 1):
                L_matrix[:, k] = eval_genlaguerre(k, alpha, x_eval)
            
            poly_vals = np.polyval(poly, x_eval)
            try:
                return np.linalg.solve(L_matrix, poly_vals)
            except:
                return None

        def cost(r):
            if np.any(r <= 0) or not np.all(np.diff(r) > 1e-6):
                return 1e10
            c = get_coeffs(r)
            if c is None:
                return 1e10
            return np.sum(np.abs(c))
            
        # Try to optimize
        x0 = roots.copy()
        try:
            result = minimize(cost, x0, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if time.time() - t0 > DEADLINE:
                raise TimeoutError()
                
            if result.fun < cost(x0):
                x_opt = np.sort(result.x)
                x_opt = np.maximum(x_opt, 1e-6)
                return list(x_opt)
            else:
                return list(x0)
                
        except Exception:
            return list(x0)
            
    except Exception:
        return list(seed_roots)
