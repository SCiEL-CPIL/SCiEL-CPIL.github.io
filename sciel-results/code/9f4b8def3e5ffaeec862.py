import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    t_start = time.time()
    time_limit = 600.0
    margin = 20.0
    
    # Pre-compute the sofa grid
    i_arr = np.arange(450)
    j_arr = np.arange(300)
    I, J = np.meshgrid(i_arr, j_arr, indexing='ij')
    x_grid = -3 + 0.01 * I
    y_grid = -1.5 + 0.01 * J
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    n_points = x_flat.size  # 135000
    
    # Pre-compute rotations for 91 angles
    a_vals = np.arange(91)
    rad_vals = np.deg2rad(a_vals)
    cos_vals = np.cos(rad_vals)
    sin_vals = np.sin(rad_vals)
    
    def compute_score(xs, ys):
        """xs, ys: arrays of length 91, positions at each angle"""
        mask = np.ones(n_points, dtype=bool)
        
        for k in range(91):
            cx = xs[k]
            cy = ys[k]
            ca = cos_vals[k]
            sa = sin_vals[k]
            
            dx = x_flat - cx
            dy = y_flat - cy
            
            u = dx * ca + dy * sa
            v = -dx * sa + dy * ca
            
            cond1 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            cond2 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            valid = cond1 | cond2
            
            mask &= valid
            
            if not np.any(mask):
                return 0.0
        
        return np.sum(mask) * 0.0001
    
    # Initial path: try a few heuristics
    # Heuristic 1: Linear interpolation
    def make_linear_path():
        xs = np.zeros(91)
        ys = np.zeros(91)
        xs[0] = 0.0
        ys[0] = 0.0
        xs[90] = -2.0
        ys[90] = 0.0
        for k in range(1, 90):
            t = k / 90.0
            xs[k] = -2.0 * t
            ys[k] = 0.0
        return xs, ys
    
    # Heuristic 2: Arc path - go up then around
    def make_arc_path():
        xs = np.zeros(91)
        ys = np.zeros(91)
        xs[0] = 0.0
        ys[0] = 0.0
        xs[90] = -2.0
        ys[90] = 0.0
        for k in range(1, 90):
            t = k / 90.0
            # Try a path that goes up in y then comes back down
            xs[k] = -2.0 * t
            ys[k] = 1.5 * np.sin(np.pi * t)
        return xs, ys
    
    # Heuristic 3: Different arc
    def make_arc_path2():
        xs = np.zeros(91)
        ys = np.zeros(91)
        xs[0] = 0.0
        ys[0] = 0.0
        xs[90] = -2.0
        ys[90] = 0.0
        for k in range(1, 90):
            t = k / 90.0
            xs[k] = -2.0 * t
            ys[k] = 2.0 * np.sin(np.pi * t)
        return xs, ys
    
    # Heuristic 4: Path that goes more vertically
    def make_arc_path3():
        xs = np.zeros(91)
        ys = np.zeros(91)
        xs[0] = 0.0
        ys[0] = 0.0
        xs[90] = -2.0
        ys[90] = 0.0
        for k in range(1, 90):
            t = k / 90.0
            xs[k] = -2.0 * t**0.5
            ys[k] = 1.0 * np.sin(np.pi * t)
        return xs, ys
    
    # Heuristic 5: Symmetric-ish path
    def make_arc_path4():
        xs = np.zeros(91)
        ys = np.zeros(91)
        xs[0] = 0.0
        ys[0] = 0.0
        xs[90] = -2.0
        ys[90] = 0.0
        for k in range(1, 90):
            t = k / 90.0
            xs[k] = -2.0 * t
            ys[k] = 0.8 * np.sin(np.pi * t)
        return xs, ys
    
    # Heuristic 6: Go up more, then swing
    def make_arc_path5():
        xs = np.zeros(91)
        ys = np.zeros(91)
        xs[0] = 0.0
        ys[0] = 0.0
        xs[90] = -2.0
        ys[90] = 0.0
        for k in range(1, 90):
            t = k / 90.0
            xs[k] = -1.5 * t - 0.5 * np.sin(np.pi * t)
            ys[k] = 1.2 * np.sin(np.pi * t)
        return xs, ys
    
    paths = [
        make_linear_path(),
        make_arc_path(),
        make_arc_path2(),
        make_arc_path3(),
        make_arc_path4(),
        make_arc_path5(),
    ]
    
    best_xs = None
    best_ys = None
    best_score = -1.0
    
    for xs_init, ys_init in paths:
        if time.time() - t_start > time_limit - margin:
            break
        s = compute_score(xs_init, ys_init)
        if s > best_score:
            best_score = s
            best_xs = xs_init.copy()
            best_ys = ys_init.copy()
    
    if best_xs is None:
        best_xs, best_ys = paths[0]
        best_score = compute_score(best_xs, best_ys)
    
    # Coordinate descent refinement
    def refine(xs, ys, max_time=None):
        if max_time is None:
            max_time = time_limit - margin
        improved = True
        passes = 0
        while improved and time.time() - t_start < max_time:
            improved = False
            passes += 1
            for k in range(1, 90):
                if time.time() - t_start > max_time:
                    break
                cx = xs[k]
                cy = ys[k]
                best_cx = cx
                best_cy = cy
                best_s = compute_score(xs, ys)
                
                # Try perturbations in x and y
                for dx in [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]:
                    if time.time() - t_start > max_time:
                        break
                    xs[k] = cx + dx
                    s = compute_score(xs, ys)
                    if s > best_s:
                        best_s = s
                        best_cx = cx + dx
                        best_cy = cy
                        improved = True
                for dy in [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]:
                    if time.time() - t_start > max_time:
                        break
                    xs[k] = cx
                    ys[k] = cy + dy
                    s = compute_score(xs, ys)
                    if s > best_s:
                        best_s = s
                        best_cx = cx
                        best_cy = cy + dy
                        improved = True
                # Also try combined
                for dx in [-0.1, 0.1]:
                    for dy in [-0.1, 0.1]:
                        if time.time() - t_start > max_time:
                            break
                        xs[k] = cx + dx
                        ys[k] = cy + dy
                        s = compute_score(xs, ys)
                        if s > best_s:
                            best_s = s
                            best_cx = cx + dx
                            best_cy = cy + dy
                            improved = True
                
                xs[k] = best_cx
                ys[k] = best_cy
        
        return xs, ys, best_score
    
    # Do several passes of refinement
    best_xs, best_ys, best_score = refine(best_xs, best_ys)
    
    # Try finer steps in a second round
    def refine_fine(xs, ys, max_time=None):
        if max_time is None:
            max_time = time_limit - margin
        improved = True
        while improved and time.time() - t_start < max_time:
            improved = False
            for k in range(1, 90):
                if time.time() - t_start > max_time:
                    break
                cx = xs[k]
                cy = ys[k]
                best_cx = cx
                best_cy = cy
                best_s = compute_score(xs, ys)
                
                for dx in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]:
                    if time.time() - t_start > max_time:
                        break
                    xs[k] = cx + dx
                    s = compute_score(xs, ys)
                    if s > best_s:
                        best_s = s
                        best_cx = cx + dx
                        best_cy = cy
                        improved = True
                for dy in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]:
                    if time.time() - t_start > max_time:
                        break
                    xs[k] = cx
                    ys[k] = cy + dy
                    s = compute_score(xs, ys)
                    if s > best_s:
                        best_s = s
                        best_cx = cx
                        best_cy = cy + dy
                        improved = True
                
                xs[k] = best_cx
                ys[k] = best_cy
        
        return xs, ys, best_score
    
    if time.time() - t_start < time_limit - margin:
        best_xs, best_ys, best_score = refine_fine(best_xs, best_ys)
    
    # Even finer
    def refine_very_fine(xs, ys, max_time=None):
        if max_time is None:
            max_time = time_limit - margin
        improved = True
        while improved and time.time() - t_start < max_time:
            improved = False
            for k in range(1, 90):
                if time.time() - t_start > max_time:
                    break
                cx = xs[k]
                cy = ys[k]
                best_cx = cx
                best_cy = cy
                best_s = compute_score(xs, ys)
                
                for dx in [-0.01, -0.005, 0.005, 0.01]:
                    if time.time() - t_start > max_time:
                        break
                    xs[k] = cx + dx
                    s = compute_score(xs, ys)
                    if s > best_s:
                        best_s = s
                        best_cx = cx + dx
                        best_cy = cy
                        improved = True
                for dy in [-0.01, -0.005, 0.005, 0.01]:
                    if time.time() - t_start > max_time:
                        break
                    xs[k] = cx
                    ys[k] = cy + dy
                    s = compute_score(xs, ys)
                    if s > best_s:
                        best_s = s
                        best_cx = cx
                        best_cy = cy + dy
                        improved = True
                
                xs[k] = best_cx
                ys[k] = best_cy
        
        return xs, ys, best_score
    
    if time.time() - t_start < time_limit - margin:
        best_xs, best_ys, best_score = refine_very_fine(best_xs, best_ys)
    
    # Final check
    if time.time() - t_start < time_limit - margin:
        final_score = compute_score(best_xs, best_ys)
        if final_score > best_score:
            best_score = final_score
    
    # Construct output
    path_pts = np.column_stack((best_xs, best_ys, a_vals))
    
    # Ensure finite
    if not np.all(np.isfinite(path_pts)):
        # Fallback to linear path
        path_pts = np.column_stack((np.linspace(0, -2, 91), np.zeros(91), a_vals))
    
    return path_pts
