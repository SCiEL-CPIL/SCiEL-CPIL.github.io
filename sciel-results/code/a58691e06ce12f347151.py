"""
Strategy: Global Optimization + Grid Search + Newton Polish.

The previous strategy relied on local Newton's method starting from x0. 
If x0 is far from the root or in a "bad" basin, Newton fails. 
This strategy treats the root-finding problem as minimizing ||F(x)||^2.
It uses a coarse grid search to find a good starting point, then 
Nelder-Mead (derivative-free) to get close to the root, and finally 
Newton's method (with finite differences) to polish to 1e-8 precision.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Global state to track time across calls (though each call is independent, 
# we use a local deadline guard to be safe within the 120s limit).
# The evaluator kills after 150s, so we aim to finish each call in < 20s.
_CALL_DEADLINE = 20.0

def _newton_polish(F, x, max_iter=50, tol=1e-12):
    """
    Newton's method with finite difference Jacobian.
    Returns (x, norm_F, success).
    """
    x = np.array(x, dtype=float)
    n_iter = 0
    for _ in range(max_iter):
        n_iter += 1
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            return x, np.inf, False
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= tol:
            return x, norm_Fx, True
        
        # Compute Jacobian
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = F(x + e)
            if not np.all(np.isfinite(Fx_plus)):
                return x, norm_Fx, False
            J[:, j] = (Fx_plus - Fx) / h
        
        try:
            dx = np.linalg.solve(J, Fx)
            x_new = x - dx
            if not np.all(np.isfinite(x_new)):
                return x, norm_Fx, False
            x = x_new
        except np.linalg.LinAlgError:
            return x, norm_Fx, False
            
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        return x, np.inf, False
    return x, np.linalg.norm(Fx), np.linalg.norm(Fx) <= tol

def find_root_system(F, x0):
    """
    Finds a root x such that ||F(x)|| <= 1e-8.
    """
    start_time = time.time()
    x0 = np.array(x0, dtype=float)
    
    # 1. Check if x0 is already a root
    Fx0 = F(x0)
    if np.all(np.isfinite(Fx0)) and np.linalg.norm(Fx0) <= 1e-8:
        return x0
    
    # 2. Grid Search for a better initial point
    # We search in a box around x0. The box size is adaptive based on ||F(x0)||
    # but we use a fixed reasonable range for 2D systems.
    # Let's try a grid of 10x10 points in a box of size 10 centered at x0.
    # If that fails, we expand.
    
    best_x = x0
    best_norm = np.linalg.norm(Fx0) if np.all(np.isfinite(Fx0)) else np.inf
    
    # Define grid parameters
    # We'll try a few scales: 1, 5, 10, 50
    scales = [1.0, 5.0, 10.0, 50.0, 100.0]
    n_grid = 5  # 5x5 grid = 25 points per scale
    
    for scale in scales:
        if time.time() - start_time > _CALL_DEADLINE * 0.5:
            break
            
        # Create grid
        xs = np.linspace(x0[0] - scale, x0[0] + scale, n_grid)
        ys = np.linspace(x0[0] - scale, x0[0] + scale, n_grid)
        
        for xi in xs:
            for yi in ys:
                x_try = np.array([xi, yi])
                try:
                    Fx_try = F(x_try)
                    if np.all(np.isfinite(Fx_try)):
                        norm_try = np.linalg.norm(Fx_try)
                        if norm_try < best_norm:
                            best_norm = norm_try
                            best_x = x_try
                except Exception:
                    continue
        
        # If we found a very good point, stop expanding
        if best_norm < 1e-4:
            break

    # 3. Nelder-Mead Optimization on ||F(x)||^2
    # This is robust to non-linearity and doesn't require derivatives.
    def objective(x):
        try:
            fx = F(x)
            if not np.all(np.isfinite(fx)):
                return 1e100
            return np.dot(fx, fx)
        except Exception:
            return 1e100

    # Start Nelder-Mead from best_x
    # Also try starting from x0 in case best_x was a local min of F but not F^2? 
    # Actually, best_x is the min of ||F||^2 found by grid.
    # Let's run Nelder-Mead from best_x.
    
    x_opt = best_x
    try:
        res = minimize(objective, x_opt, method='Nelder-Mead', 
                       options={'xatol': 1e-10, 'fatol': 1e-20, 'maxiter': 200, 'maxfev': 500})
        if res.success and res.fun < best_norm:
            x_opt = res.x
            best_norm = np.sqrt(res.fun)
    except Exception:
        pass

    # 4. Newton Polish
    # Use x_opt as the starting point for Newton.
    x_final, norm_final, success = _newton_polish(F, x_opt, max_iter=30, tol=1e-12)
    
    if success:
        return x_final

    # If Newton failed, try Newton from x0 as well, just in case
    x_final2, norm_final2, success2 = _newton_polish(F, x0, max_iter=30, tol=1e-12)
    if success2:
        return x_final2

    # If both failed, return the best point we found
    if norm_final < norm_final2:
        return x_final
    else:
        return x_final2
# EVOLVE-BLOCK-END
