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
        """
        Compute score for a path defined by xs, ys (length 91).
        Uses chunking to reduce memory reads of x_flat/y_flat.
        """
        mask = np.ones(n_points, dtype=bool)
        chunk_size = 10
        
        for start_k in range(0, 91, chunk_size):
            end_k = min(start_k + chunk_size, 91)
            k_slice = slice(start_k, end_k)
            
            xs_chunk = xs[k_slice]  # shape (chunk_size,)
            ys_chunk = ys[k_slice]
            cos_chunk = cos_vals[k_slice]
            sin_chunk = sin_vals[k_slice]
            
            # dx shape: (chunk_size, n_points)
            # This reads x_flat once for the chunk
            dx = x_flat[None, :] - xs_chunk[:, None]
            dy = y_flat[None, :] - ys_chunk[:, None]
            
            u = dx * cos_chunk[:, None] + dy * sin_chunk[:, None]
            v = -dx * sin_chunk[:, None] + dy * cos_chunk[:, None]
            
            cond1 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            cond2 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            valid_chunk = cond1 | cond2  # shape (chunk_size, n_points)
            
            valid_all = valid_chunk.all(axis=0)  # shape (n_points,)
            mask &= valid_all
            
            if not np.any(mask):
                return 0.0
        
        return np.sum(mask) * 0.0001

    # Heuristics
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

    def make_arc_path():
        xs = np.zeros(91)
        ys = np.zeros(91)
        xs[0] = 0.0
        ys[0] = 0.0
        xs[90] = -2.0
        ys[90] = 0.0
        for k in range(1, 90):
            t = k / 90.0
            xs[k] = -2.0 * t
            ys[k] = 1.5 * np.sin(np.pi * t)
        return xs, ys
        
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

    def refine(xs, ys, steps, max_time=None):
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
                
                # Try perturbations in x
                for dx in steps:
                    xs[k] = cx + dx
                    s = compute_score(xs, ys)
                    if s > best_s:
                        best_s = s
                        best_cx = cx + dx
                        best_cy = cy
                        improved = True
                # Try perturbations in y
                for dy in steps:
                    xs[k] = cx
                    ys[k] = cy + dy
                    s = compute_score(xs, ys)
                    if s > best_s:
                        best_s = s
                        best_cx = cx
                        best_cy = cy + dy
                        improved = True
                # Try combined
                for dx in steps:
                    for dy in steps:
                        if abs(dx) > 0.1 and abs(dy) > 0.1: continue # Skip large diagonals
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
        
        return xs, ys, best_s

    # Coarse Refinement
    coarse_steps = [-0.5, -0.2, -0.1, 0.1, 0.2, 0.5]
    best_xs, best_ys, best_score = refine(best_xs, best_ys, coarse_steps)

    # Medium Refinement
    if time.time() - t_start < time_limit - margin:
        med_steps = [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]
        best_xs, best_ys, best_score = refine(best_xs, best_ys, med_steps)

    # Fine Refinement
    if time.time() - t_start < time_limit - margin:
        fine_steps = [-0.1, -0.05, -0.02, 0.02, 0.05, 0.1]
        best_xs, best_ys, best_score = refine(best_xs, best_ys, fine_steps)

    # Very Fine Refinement
    if time.time() - t_start < time_limit - margin:
        vfine_steps = [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]
        best_xs, best_ys, best_score = refine(best_xs, best_ys, vfine_steps)

    # Ultra Fine Refinement
    if time.time() - t_start < time_limit - margin:
        ufine_steps = [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]
        best_xs, best_ys, best_score = refine(best_xs, best_ys, ufine_steps)

    # Random Restart / Perturbation if time remains
    if time.time() - t_start < time_limit - 60:
        rng = np.random.default_rng(42)
        for _ in range(5): # Try a few restarts
            if time.time() - t_start > time_limit - margin:
                break
            # Perturb the best path
            pert_xs = best_xs + rng.normal(0, 0.1, 91)
            pert_ys = best_ys + rng.normal(0, 0.1, 91)
            # Keep endpoints fixed
            pert_xs[0] = 0.0
            pert_ys[0] = 0.0
            pert_xs[90] = -2.0
            pert_ys[90] = 0.0
            
            # Quick refine
            pert_xs, pert_ys, pert_score = refine(pert_xs, pert_ys, [-0.05, 0.05])
            
            if pert_score > best_score:
                best_score = pert_score
                best_xs = pert_xs
                best_ys = pert_ys

    # Final check
    if time.time() - t_start < time_limit - margin:
        final_score = compute_score(best_xs, best_ys)
        if final_score > best_score:
            best_score = final_score

    # Construct output
    path_pts = np.column_stack((best_xs, best_ys, a_vals))
    
    # Ensure finite
    if not np.all(np.isfinite(path_pts)):
        path_pts = np.column_stack((np.linspace(0, -2, 91), np.zeros(91), a_vals))
    
    return path_pts
