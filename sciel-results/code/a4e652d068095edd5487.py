import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_total = 0.0
    best_xy = None
    best_r = None
    best_valid = False

    def _eval_packing(xy, r):
        """
        Evaluate a packing: returns (total_radius, is_valid).
        """
        n = len(xy)
        for i in range(n):
            x, y, ri = xy[i, 0], xy[i, 1], r[i]
            if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
                return 0.0, False
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - 1e-9:
                    return 0.0, False
        return np.sum(r), True

    def _penalized_objective(params, n, penalty_weight=1000.0):
        """
        Objective function for optimization: minimize negative sum of radii + penalties.
        params: flat array of shape (3*n,) containing [x_0, y_0, r_0, x_1, y_1, r_1, ...]
        """
        params = np.asarray(params)
        xy = params[0::3].reshape(-1, 2)
        r = params[1::3].reshape(-1)
        
        # Clip radii to be positive
        r = np.maximum(r, 1e-5)
        
        # Wall violations (soft)
        wall_violation = 0.0
        for i in range(n):
            x, y, ri = xy[i, 0], xy[i, 1], r[i]
            wall_violation += max(0.0, ri - x)**2
            wall_violation += max(0.0, ri - (1.0 - x))**2
            wall_violation += max(0.0, ri - y)**2
            wall_violation += max(0.0, ri - (1.0 - y))**2
        
        # Overlap violations (soft)
        overlap_violation = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap_violation += (min_dist - dist)**2
        
        penalty = penalty_weight * (wall_violation + overlap_violation)
        total_radius = np.sum(r)
        
        return -total_radius + penalty

    def _grid_init(side):
        """Create a grid of up to N circles."""
        positions = []
        for i in range(side):
            for j in range(side):
                if len(positions) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.001
                    positions.append([x, y, r])
        while len(positions) < N:
            positions.append([0.5, 0.5, 0.01])
        return np.array(positions[:N])

    def _random_init():
        """Create a random initialization."""
        positions = []
        for _ in range(N):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = np.random.uniform(0.01, 0.1)
            positions.append([x, y, r])
        return np.array(positions)

    def _greedy_init():
        """Greedy insertion heuristic."""
        positions = []
        # Start with a few large circles
        for _ in range(5):
            x = np.random.uniform(0.2, 0.8)
            y = np.random.uniform(0.2, 0.8)
            r = 0.1
            positions.append([x, y, r])
        
        # Add remaining circles greedily
        for _ in range(N - 5):
            best_pos = None
            best_r = 0.0
            # Try random positions
            for _ in range(100):
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                # Calculate max radius
                min_dist = min(x, 1-x, y, 1-y)
                for px, py, pr in positions:
                    dist = np.sqrt((x-px)**2 + (y-py)**2)
                    min_dist = min(min_dist, dist - pr)
                if min_dist > best_r:
                    best_r = min_dist
                    best_pos = [x, y, best_r]
            if best_pos is not None and best_r > 0.001:
                positions.append(best_pos)
            else:
                # Fallback
                positions.append([0.5, 0.5, 0.01])
        
        return np.array(positions[:N])

    def _best_fit_local_search(xy, r, deadline, grid_size=80):
        """
        Iteratively update each circle to the best position (maximizing its radius)
        given all other circles fixed.
        """
        n = len(xy)
        best_total, best_valid = _eval_packing(xy, r)
        best_xy = xy.copy()
        best_r = r.copy()
        if best_valid:
            best_total = np.sum(best_r)
        else:
            best_total = 0.0

        # Candidate grid
        gx = np.linspace(0.0, 1.0, grid_size)
        gy = np.linspace(0.0, 1.0, grid_size)
        GX, GY = np.meshgrid(gx, gy, indexing='ij')
        candidates = np.column_stack([GX.ravel(), GY.ravel()])
        M = len(candidates)

        # Precompute candidate wall clearances
        wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                                np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

        max_sweeps = 15
        for sweep in range(max_sweeps):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                others_xy = xy[mask]
                others_r = r[mask]

                if len(others_xy) > 0:
                    dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                    dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                    dists = np.sqrt(dx * dx + dy * dy)
                    circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                    r_candidates = np.minimum(wall_clear, circle_clear)
                else:
                    r_candidates = wall_clear.copy()

                idx = int(np.argmax(r_candidates))
                new_x, new_y = candidates[idx, 0], candidates[idx, 1]
                new_r = r_candidates[idx]

                old_r = r[i]
                if new_r > old_r + 1e-10:
                    xy[i, 0] = new_x
                    xy[i, 1] = new_y
                    r[i] = new_r
                    improved = True

            total, valid = _eval_packing(xy, r)
            if valid and total > best_total + 1e-10:
                best_total = total
                best_valid = True
                best_xy = xy.copy()
                best_r = r.copy()

            if not improved:
                break

        return best_xy, best_r, best_total, best_valid

    def _gradient_optimize(xy, r, deadline, max_iter=200, penalty_weight=1000.0):
        """
        Use L-BFGS-B with penalty terms to optimize the packing.
        """
        n = len(xy)
        params = np.zeros(3 * n)
        params[0::3] = xy[:, 0]
        params[1::3] = xy[:, 1]
        params[2::3] = r
        
        bounds = []
        for i in range(n):
            bounds.append((0.001, 0.999))
            bounds.append((0.001, 0.999))
            bounds.append((0.001, 0.4))
        
        best_params = params.copy()
        best_total = 0.0
        best_valid = False
        
        init_total, init_valid = _eval_packing(xy, r)
        if init_valid:
            best_total = init_total
            best_valid = True
        
        try:
            result = minimize(
                _penalized_objective,
                params,
                args=(n, penalty_weight),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-8}
            )
            
            opt_params = result.x
            opt_xy = opt_params[0::3].reshape(-1, 2)
            opt_r = np.maximum(opt_params[1::3], 0.001)
            
            total, valid = _eval_packing(opt_xy, opt_r)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_params = opt_params.copy()
                
        except Exception:
            pass
        
        # Polish
        if time.time() < deadline - 5.0 and best_valid:
            try:
                result2 = minimize(
                    _penalized_objective,
                    best_params,
                    args=(n, penalty_weight * 5),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-16, 'gtol': 1e-10}
                )
                opt_params2 = result2.x
                opt_xy2 = opt_params2[0::3].reshape(-1, 2)
                opt_r2 = np.maximum(opt_params2[1::3], 0.001)
                
                total2, valid2 = _eval_packing(opt_xy2, opt_r2)
                if valid2 and total2 > best_total:
                    best_total = total2
                    best_valid = True
                    best_params = opt_params2.copy()
                    
            except Exception:
                pass
        
        final_xy = best_params[0::3].reshape(-1, 2)
        final_r = np.maximum(best_params[1::3], 0.001)
        
        return final_xy, final_r, best_total, best_valid

    # Run multiple strategies
    strategies = [
        lambda: _grid_init(5),
        lambda: _grid_init(6),
        lambda: _random_init(),
        lambda: _greedy_init(),
    ]
    
    for _ in range(4): # 4 restarts
        if time.time() > deadline - 20.0:
            break
        
        for strat in strategies:
            if time.time() > deadline - 10.0:
                break
            
            try:
                init_pos = strat()
                xy = init_pos[:, :2].copy()
                r = init_pos[:, 2].copy()
                
                # Step 1: Local Search
                out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline - 10.0, grid_size=60)
                if valid and total > best_total:
                    best_total = total
                    best_valid = True
                    best_xy = out_xy
                    best_r = out_r
                
                # Step 2: Gradient Optimization
                if time.time() < deadline - 5.0 and best_valid:
                    g_xy, g_r, g_total, g_valid = _gradient_optimize(
                        best_xy, best_r, deadline - 5.0, max_iter=150, penalty_weight=1000.0
                    )
                    if g_valid and g_total > best_total:
                        best_total = g_total
                        best_valid = True
                        best_xy = g_xy
                        best_r = g_r
                        
            except Exception:
                continue

    # Final fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
    
    return result
