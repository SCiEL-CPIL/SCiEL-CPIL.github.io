"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def construct():
    deadline = time.time() + 115.0  # Safety margin before 150s limit
    best_result = None
    best_score = 0.0
    best_w = 1.0

    def unpack(vars_):
        w = vars_[0]
        x = vars_[1::3]
        y = vars_[2::3]
        r = vars_[3::3]
        return w, x, y, r

    def validate(x, y, r, w, h):
        if np.any(r < -1e-9):
            return False
        if np.any(x < r - 1e-9) or np.any(x > w - r + 1e-9):
            return False
        if np.any(y < r - 1e-9) or np.any(y > h - r + 1e-9):
            return False
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return False
        return True

    def grid_init(w, h, seed, cols, rows):
        rng = np.random.default_rng(seed)
        
        xs = np.zeros(N)
        ys = np.zeros(N)
        
        # Fill grid
        idx = 0
        for r_idx in range(rows):
            for c_idx in range(cols):
                if idx >= N:
                    break
                # Center of cell
                x = (c_idx + 0.5) * w / cols
                y = (r_idx + 0.5) * h / rows
                
                # Add jitter
                jitter = 0.1 * min(w/cols, h/rows)
                x += rng.uniform(-jitter, jitter)
                y += rng.uniform(-jitter, jitter)
                
                # Clamp
                x = np.clip(x, 0.01, w - 0.01)
                y = np.clip(y, 0.01, h - 0.01)
                
                xs[idx] = x
                ys[idx] = y
                idx += 1
        
        # Initial radii: small
        # Estimate min distance
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                min_dist = min(min_dist, d)
        for i in range(N):
            min_dist = min(min_dist, xs[i], w-xs[i], ys[i], h-ys[i])
        
        rs = np.full(N, 0.4 * min_dist)
        
        return xs, ys, rs

    def run_local_optimization(x0, bounds, maxiter=300):
        """
        Performs a local optimization using a soft-body penalty approach.
        The penalty weight increases dynamically to enforce constraints.
        """
        current_x = x0.copy()
        best_local_x = current_x.copy()
        best_local_score = -np.inf
        
        # Dynamic penalty weight schedule
        penalty_weights = [10, 100, 1000, 5000, 20000]
        
        for penalty_weight in penalty_weights:
            if time.time() > deadline - 2:
                break
                
            def objective(vars_):
                w, x, y, r = unpack(vars_)
                h = 2.0 - w
                
                # Sum of radii to maximize (minimize negative)
                sum_r = np.sum(r)
                
                # Overlap Penalty
                overlap_pen = 0.0
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d = np.sqrt(dx*dx + dy*dy)
                        overlap = r[i] + r[j] - d
                        if overlap > 0:
                            overlap_pen += overlap * overlap
                
                # Wall Penalty
                wall_pen = 0.0
                for i in range(N):
                    v1 = r[i] - x[i]
                    if v1 > 0: wall_pen += v1 * v1
                    v2 = r[i] + x[i] - w
                    if v2 > 0: wall_pen += v2 * v2
                    v3 = r[i] - y[i]
                    if v3 > 0: wall_pen += v3 * v3
                    v4 = r[i] + y[i] - h
                    if v4 > 0: wall_pen += v4 * v4

                return -sum_r + penalty_weight * (overlap_pen + wall_pen)

            try:
                result = minimize(
                    objective, current_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': 1e-9}
                )
                
                # Update current state to result for next penalty weight
                current_x = result.x
                
                # Check validity of the current result
                w, x, y, r = unpack(current_x)
                h = 2.0 - w
                if validate(x, y, r, w, h):
                    score = np.sum(r)
                    if score > best_local_score:
                        best_local_score = score
                        best_local_x = current_x.copy()
                        
            except Exception:
                continue
        
        return best_local_x, best_local_score

    # Bounds for L-BFGS-B
    # w in [0.5, 1.5]
    # x, y, r > 0
    lb_base = np.zeros(1 + 3*N)
    ub_base = np.zeros(1 + 3*N)
    lb_base[0] = 0.5
    ub_base[0] = 1.5
    for i in range(N):
        lb_base[1 + 3*i] = 0.0 # x
        ub_base[1 + 3*i] = 2.0
        lb_base[2 + 3*i] = 0.0 # y
        ub_base[2 + 3*i] = 2.0
        lb_base[3 + 3*i] = 0.0 # r
        ub_base[3 + 3*i] = 1.0
    bounds = list(zip(lb_base, ub_base))

    # Strategy:
    # 1. Try various aspect ratios and grid topologies.
    # 2. Use soft-body optimization to refine.
    
    # Grid configurations (cols, rows) that can hold 21 circles
    # 5x5 = 25 (good)
    # 4x6 = 24 (good)
    # 3x7 = 21 (exact)
    # 7x3 = 21 (exact)
    # 6x4 = 24 (good)
    # 2x11 = 22 (bad aspect)
    
    grid_configs = [
        (5, 5),
        (4, 6),
        (6, 4),
        (3, 7),
        (7, 3),
        (5, 4), # 20, one empty
        (4, 5), # 20, one empty
    ]
    
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.25, 1.4]
    
    # We need to be careful with time.
    # 7 grids * 7 aspects * 2 restarts = 98 optimizations.
    # If each takes ~1s, that's ~100s. Tight but doable.
    # Let's limit restarts to 1 initially and increase if time permits.
    
    restarts_per_config = 2
    
    for w_init in aspect_ratios:
        if time.time() > deadline - 10:
            break
        h_init = 2.0 - w_init
        
        # Filter grids that roughly match aspect ratio
        # w/h approx cols/rows
        target_ar = w_init / h_init
        
        for cols, rows in grid_configs:
            if time.time() > deadline - 5:
                break
            
            grid_ar = cols / rows
            # Skip if aspect ratios are too mismatched
            if abs(target_ar - grid_ar) / grid_ar > 0.4:
                continue
                
            for restart in range(restarts_per_config):
                if time.time() > deadline - 2:
                    break
                    
                seed = restart * 1000 + int(w_init * 100) + cols * 10 + rows
                xs, ys, rs = grid_init(w_init, h_init, seed, cols, rows)
                
                # Variable vector: [w, x0, y0, r0, x1, y1, r1, ...]
                x0 = np.zeros(1 + 3*N)
                x0[0] = w_init
                x0[1::3] = xs
                x0[2::3] = ys
                x0[3::3] = rs
                
                # Run the local optimizer
                try:
                    best_x, score = run_local_optimization(x0, bounds)
                    
                    if score > best_score:
                        w, x, y, r = unpack(best_x)
                        h = 2.0 - w
                        if validate(x, y, r, w, h):
                            best_score = score
                            best_result = np.column_stack([x, y, r])
                            best_w = w
                            
                except Exception:
                    continue

    # If we still have time, refine the best solution further
    if best_result is not None and time.time() < deadline - 5:
        # Reconstruct x0 from best_result
        x0 = np.zeros(1 + 3*N)
        x0[0] = best_w
        x0[1::3] = best_result[:, 0]
        x0[2::3] = best_result[:, 1]
        x0[3::3] = best_result[:, 2]
        
        # Run a final high-precision refinement
        # Use a very high penalty weight and more iterations
        def final_objective(vars_):
            w, x, y, r = unpack(vars_)
            h = 2.0 - w
            sum_r = np.sum(r)
            overlap_pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    overlap = r[i] + r[j] - d
                    if overlap > 0:
                        overlap_pen += overlap * overlap
            wall_pen = 0.0
            for i in range(N):
                v1 = r[i] - x[i]
                if v1 > 0: wall_pen += v1 * v1
                v2 = r[i] + x[i] - w
                if v2 > 0: wall_pen += v2 * v2
                v3 = r[i] - y[i]
                if v3 > 0: wall_pen += v3 * v3
                v4 = r[i] + y[i] - h
                if v4 > 0: wall_pen += v4 * v4
            return -sum_r + 100000 * (overlap_pen + wall_pen)

        try:
            result = minimize(
                final_objective, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            w, x, y, r = unpack(result.x)
            h = 2.0 - w
            if validate(x, y, r, w, h):
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([x, y, r])
                    best_w = w
        except Exception:
            pass

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells)

    return best_result
# EVOLVE-BLOCK-END
