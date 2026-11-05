import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh_tridiagonal
from scipy.special import roots_laguerre

def construct():
    """
    Optimizes the off-diagonal entries of a 12x12 tridiagonal matrix
    to minimize the Cohn-Goncalves uncertainty bound C4.
    """
    N = 12
    # Fixed diagonal entries based on Laguerre recurrence
    d = np.array([2*i + 1 for i in range(N)], dtype=float)
    o_init = np.array([i for i in range(1, N)], dtype=float)
    
    def c4_bound(roots):
        """Compute the C4 upper bound from root locations."""
        # C4 = sqrt(sum(r_i)) / (prod(r_i))^(1/24)
        # This is a common form for uncertainty bounds
        sum_r = np.sum(roots)
        prod_r = np.prod(roots)
        if prod_r <= 0 or sum_r <= 0:
            return 1e10
        return np.sqrt(sum_r) / (prod_r ** (1.0/24.0))
    
    def cost(params):
        o = params
        try:
            w, _ = eigh_tridiagonal(d, o, select='all')
        except Exception:
            return 1e10
            
        if np.any(w <= 0) or np.any(np.diff(w) <= 0):
            return 1e10 + np.sum(w[w <= 0])**2
            
        return c4_bound(w)

    # Try multiple initial conditions
    best_cost = np.inf
    best_w = None
    
    # Initial condition 1: Standard Laguerre off-diagonals
    x0_list = [o_init.copy()]
    
    # Initial condition 2: Scaled off-diagonals
    x0_list.append(o_init * 0.5)
    x0_list.append(o_init * 1.5)
    
    # Initial condition 3: Constant off-diagonals
    x0_list.append(np.ones(N-1) * 5.0)
    x0_list.append(np.ones(N-1) * 10.0)
    
    bounds = [(0.01, 200.0) for _ in range(N-1)]
    
    for x0 in x0_list:
        try:
            res = minimize(cost, x0, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.success and res.fun < best_cost:
                best_cost = res.fun
                w, _ = eigh_tridiagonal(d, res.x, select='all')
                if np.all(w > 0) and np.all(np.diff(w) > 0):
                    best_w = w
        except Exception:
            continue
    
    # Also try Nelder-Mead for robustness
    for x0 in x0_list[:2]:
        try:
            res = minimize(cost, x0, method='Nelder-Mead', 
                         options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_cost:
                w, _ = eigh_tridiagonal(d, res.x, select='all')
                if np.all(w > 0) and np.all(np.diff(w) > 0):
                    best_cost = res.fun
                    best_w = w
        except Exception:
            continue
    
    if best_w is not None:
        return [float(r) for r in best_w]
    else:
        # Fallback to standard Laguerre roots
        roots, _ = roots_laguerre(12)
        return [float(r) for r in roots]
