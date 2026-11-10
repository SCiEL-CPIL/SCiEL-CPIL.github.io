import numpy as np
import time
import sys
from scipy.optimize import differential_evolution, minimize

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 595.0  # 5s safety margin
    
    N = 8
    
    # 1D Union Length calculation for a given x vector and c
    def union_length(x, c):
        # x: shape (8,)
        # c: scalar
        # Returns length of union of intervals
        starts = x + c * np.arange(N) / N
        ends = starts + 1.0 / N
        # Sort by start
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        # Merge
        total_len = 0.0
        current_end = -np.inf
        for s, e in zip(s_sorted, e_sorted):
            if s > current_end:
                total_len += (e - s)
                current_end = e
            else:
                if e > current_end:
                    total_len += (e - current_end)
                    current_end = e
        return total_len

    # Define fixed c samples - use more for better accuracy
    c_samples = np.linspace(0.0, 1.0, 500)
    
    def cost_func(x):
        # x is flattened array of 8 floats
        total = 0.0
        for c in c_samples:
            total += union_length(x, c)
        return total / len(c_samples)

    # Fallback: Keich binary construction
    def fallback_positions():
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        return np.stack([base_x, base_y], axis=1)
    
    # Optimization
    # Tighter bounds: the optimal x values should be in a narrow range
    # Since intervals have width 1/8 and shift by at most 1.0,
    # keeping them in [-0.5, 0.5] or similar should suffice
    bounds = [(-1.0, 1.0) for _ in range(N)]
    
    x_opt = None
    
    # Run differential evolution with time checking
    try:
        # Use callback to check time
        class TimeExceeded(Exception):
            pass
        
        def time_check_callback(result):
            if time.time() > deadline:
                raise TimeExceeded()
        
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                maxiter=100, 
                popsize=15, 
                tol=1e-8, 
                seed=42,
                init='latinhypercube',
                callback=time_check_callback,
                polish=False
            )
            x_opt = result.x
        except TimeExceeded:
            # Use best found so far if available
            if hasattr(result, 'x') and result.x is not None:
                x_opt = result.x
    except Exception as e:
        x_opt = None
    
    # If DE didn't produce a good result, try local optimization from multiple starts
    if x_opt is None or time.time() < deadline - 30:
        # Try Nelder-Mead from the best DE result or from random starts
        best_cost = np.inf
        best_x = None
        
        # Start from DE result if available
        if x_opt is not None:
            try:
                res = minimize(cost_func, x_opt, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_cost:
                    best_cost = res.fun
                    best_x = res.x
            except:
                pass
        
        # Try a few random starts
        rng = np.random.default_rng(123)
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            x0 = rng.uniform(-0.5, 0.5, size=N)
            try:
                res = minimize(cost_func, x0, method='Nelder-Mead', 
                             options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_cost:
                    best_cost = res.fun
                    best_x = res.x
            except:
                pass
        
        if best_x is not None:
            x_opt = best_x
    
    # If still no good result, use fallback
    if x_opt is None:
        positions = fallback_positions()
        return positions
    
    # Construct positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = x_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    # Validate
    if not np.all(np.isfinite(positions)):
        positions = fallback_positions()
        
    return positions
