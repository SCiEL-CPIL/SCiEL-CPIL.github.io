import numpy as np
import time
import sys
from scipy.optimize import differential_evolution, minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    w = 1.0 / N
    start_time = time.time()
    deadline = 630.0
    
    # Precompute indices
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten().astype(np.float64)
    j_flat = j_grid.flatten().astype(np.float64)
    
    def compute_union_area_fast(x_ints, y_ints):
        """
        Compute the area of the union of rectangles defined by x_ints and y_ints.
        Uses sweep line algorithm.
        """
        n = len(x_ints)
        if n == 0:
            return 0.0
            
        events = []
        for i in range(n):
            x_l, x_r = x_ints[i]
            y_l, y_r = y_ints[i]
            events.append((x_l, 1, y_l, y_r))
            events.append((x_r, -1, y_l, y_r))
            
        events.sort(key=lambda e: e[0])
        
        active_y_intervals = []
        area = 0.0
        last_x = events[0][0]
        
        for event in events:
            x, type, y_l, y_r = event
            if x > last_x:
                if active_y_intervals:
                    # Sort and merge y intervals
                    active_y_intervals.sort()
                    merged = []
                    cur_l, cur_r = active_y_intervals[0]
                    for y_l2, y_r2 in active_y_intervals[1:]:
                        if y_l2 <= cur_r:
                            cur_r = max(cur_r, y_r2)
                        else:
                            merged.append((cur_l, cur_r))
                            cur_l, cur_r = y_l2, y_r2
                    merged.append((cur_l, cur_r))
                    
                    total_y_len = sum(r - l for l, r in merged)
                    area += (x - last_x) * total_y_len
                    
            if type == 1:
                active_y_intervals.append((y_l, y_r))
            else:
                # Remove (y_l, y_r)
                # Use a list for simplicity, N is small
                for idx, (l, r) in enumerate(active_y_intervals):
                    if l == y_l and r == y_r:
                        active_y_intervals.pop(idx)
                        break
                        
            last_x = x
            
        return area

    def compute_volume_mc(xk, yk, num_c=200):
        """
        Monte Carlo estimate of volume using exact area at sampled c values.
        """
        c_samples = np.linspace(0.0, 1.0, num_c)
        total_area = 0.0
        
        for c in c_samples:
            x_ints = []
            y_ints = []
            for idx in range(64):
                x_l = xk[idx] + c * i_flat[idx] / 8.0
                x_r = x_l + w
                y_l = yk[idx] + c * j_flat[idx] / 8.0
                y_r = y_l + w
                x_ints.append((x_l, x_r))
                y_ints.append((y_l, y_r))
            
            area = compute_union_area_fast(x_ints, y_ints)
            total_area += area
            
        return total_area / num_c

    # Initial configuration
    alpha = 0.0625
    xk_init = (-i_flat * alpha).astype(np.float64)
    yk_init = (-j_flat * alpha).astype(np.float64)
    x0 = np.concatenate([xk_init, yk_init])
    
    def objective(params):
        if time.time() - start_time > deadline - 10:
            return 1e9
        xk = params[:64]
        yk = params[64:]
        try:
            vol = compute_volume_mc(xk, yk, 200)
            return vol
        except:
            return 1e9

    bounds = [(-1.0, 1.0)] * 128

    try:
        res_de = differential_evolution(
            objective, 
            bounds, 
            maxiter=30, 
            popsize=10, 
            tol=0.01, 
            seed=42,
            polish=False
        )
        
        best_params = res_de.x
        best_vol = res_de.fun
        
        if time.time() - start_time < deadline - 30:
            res_powell = minimize(
                objective, 
                best_params, 
                method='Powell', 
                options={'maxiter': 50, 'xtol': 1e-6, 'ftol': 1e-6}
            )
            if res_powell.fun < best_vol:
                best_params = res_powell.x
                best_vol = res_powell.fun
                
    except Exception as e:
        print(f"Optimization error: {e}")
        best_params = x0
        best_vol = objective(x0)

    xk_best = best_params[:64]
    yk_best = best_params[64:]
    
    if not np.all(np.isfinite(xk_best)) or not np.all(np.isfinite(yk_best)):
        xk_best = xk_init
        yk_best = yk_init

    positions = np.stack([xk_best, yk_best], axis=1)
    return positions
