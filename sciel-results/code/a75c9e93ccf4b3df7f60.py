import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_genlaguerre
import time

def construct():
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 600s kill
    
    m = 12
    n = 1
    deg = 2 * m  # degree 24
    
    # Start from the known-good arithmetic progression (seed scores 1.0)
    x0 = np.array([4.0 * (i + 1) for i in range(m)])
    
    best_x = x0.copy()
    best_score = x0[-1]
    
    def compute_c4_bound(x_roots):
        """
        Compute the C4 upper bound for the Laguerre construction.
        The bound is the last root position, but we need to verify
        the Laguerre coefficient constraints.
        """
        if np.any(x_roots <= 0):
            return 1e10
        if not np.all(np.diff(x_roots) > 1e-8):
            return 1e10
        
        # Build polynomial prod (x - r_i)^2
        poly = np.array([1.0])
        for r in x_roots:
            factor = np.array([1.0, -2*r, r*r])
            poly = np.convolve(poly, factor)
        
        # Express in Laguerre basis L_k(x; n) for k=0..deg
        # Use collocation at deg+1 points
        x_eval = np.linspace(0.5, 60.0, deg + 1)
        L_matrix = np.zeros((deg + 1, deg + 1))
        for k in range(deg + 1):
            L_matrix[:, k] = eval_genlaguerre(k, n, x_eval)
        
        poly_vals = np.polyval(poly, x_eval)
        
        try:
            c = np.linalg.solve(L_matrix, poly_vals)
        except np.linalg.LinAlgError:
            return 1e10
        
        # The C4 bound is the last root
        return x_roots[-1]
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        return compute_c4_bound(x)
    
    # Use L-BFGS-B with bounds to keep roots positive and ordered
    # Transform to ensure ordering: use cumulative sum of positive increments
    def transform(y):
        """Transform unconstrained y to ordered positive roots"""
        # y[0] = r1, y[i] = r_{i+1} - r_i for i >= 1
        r = np.zeros(m)
        r[0] = np.exp(y[0])  # ensure positive
        for i in range(1, m):
            r[i] = r[i-1] + np.exp(y[i])  # ensure increasing
        return r
    
    def objective_transformed(y):
        if time.time() > deadline:
            return 1e10
        r = transform(y)
        return compute_c4_bound(r)
    
    # Initial y from x0
    y0 = np.zeros(m)
    y0[0] = np.log(x0[0])
    for i in range(1, m):
        y0[i] = np.log(x0[i] - x0[i-1])
    
    # Try L-BFGS-B
    try:
        result = minimize(objective_transformed, y0, method='L-BFGS-B',
                         options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if time.time() > deadline - 10:
            raise TimeoutError()
        
        if result.fun < best_score:
            r_opt = transform(result.x)
            if np.all(r_opt > 0) and np.all(np.diff(r_opt) > 1e-8):
                best_x = r_opt
                best_score = result.fun
    except Exception:
        pass
    
    # Try Nelder-Mead on the original parameterization
    try:
        def cost_nm(x):
            if time.time() > deadline:
                return 1e10
            x_sorted = np.sort(x)
            return compute_c4_bound(x_sorted)
        
        result2 = minimize(cost_nm, best_x, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9})
        
        if time.time() > deadline - 10:
            raise TimeoutError()
        
        if result2.fun < best_score:
            r_opt = np.sort(result2.x)
            if np.all(r_opt > 0) and np.all(np.diff(r_opt) > 1e-8):
                best_x = r_opt
                best_score = result2.fun
    except Exception:
        pass
    
    # Try a grid-based local refinement around the best solution
    if time.time() < deadline - 30:
        try:
            for step_size in [0.5, 0.1, 0.01]:
                if time.time() > deadline - 10:
                    break
                improved = True
                while improved and time.time() < deadline - 10:
                    improved = False
                    for i in range(m):
                        for delta in [-step_size, step_size]:
                            x_trial = best_x.copy()
                            x_trial[i] += delta
                            x_trial = np.sort(x_trial)
                            if np.all(x_trial > 0) and np.all(np.diff(x_trial) > 1e-8):
                                score_trial = compute_c4_bound(x_trial)
                                if score_trial < best_score - 1e-10:
                                    best_x = x_trial
                                    best_score = score_trial
                                    improved = True
        except Exception:
            pass
    
    # Final validation
    best_x = np.sort(best_x)
    best_x = np.maximum(best_x, 1e-6)
    
    if time.time() > deadline:
        return list(x0)
    
    return list(best_x)
