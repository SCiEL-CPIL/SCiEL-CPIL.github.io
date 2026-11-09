"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    t_start = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    best_config = None
    best_score = -1
    
    rng = np.random.RandomState(42)
    
    # Function to compute radius vectorized
    def compute_radius(pts):
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        b_left = pts[:, 0] + 0.5
        b_right = 0.5 - pts[:, 0]
        b_bottom = pts[:, 1] + 0.5
        b_top = 0.5 - pts[:, 1]
        min_bound = np.min([np.min(b_left), np.min(b_right), np.min(b_bottom), np.min(b_top)])
        
        return min(min_pair / 2.0, min_bound)

    # Function to generate hexagonal grid
    def generate_hex_grid(n_rows, n_cols, r):
        points = []
        dx = 2 * r
        dy = r * np.sqrt(3)
        
        # Center the grid
        # Total width approx (n_cols - 1) * dx + 2r
        # Total height approx (n_rows - 1) * dy + 2r
        
        for row in range(n_rows):
            y = -0.5 + r + row * dy
            offset = 0 if row % 2 == 0 else dx / 2
            for col in range(n_cols):
                x = -0.5 + r + offset + col * dx
                # Adjust x to center better?
                # For even rows, width is (n_cols-1)*dx
                # For odd rows, width is (n_cols-1)*dx + dx/2
                # Let's just generate and then center the whole set
                points.append([x, y])
        
        points = np.array(points)
        # Center the points
        points[:, 0] -= np.mean(points[:, 0])
        points[:, 1] -= np.mean(points[:, 1])
        return points

    # Try a few different hexagonal configurations
    configs_to_try = []
    
    # Config 1: 14 rows, 14 cols (196 points, pad)
    r1 = 0.035
    grid1 = generate_hex_grid(14, 14, r1)
    if len(grid1) < N:
        # Pad with random points near edges? Or just resize?
        # Better to use a grid that has >= N points
        pass
        
    # Let's use a grid that definitely has >= 200 points
    # 14 rows x 15 cols = 210 points
    r_init = 0.034
    grid_full = generate_hex_grid(14, 15, r_init)
    
    # We have 210 points. We need 200.
    # Which 10 to remove?
    # Removing points from the corners might be good?
    # Or just take the first 200?
    # Taking first 200 breaks symmetry.
    # Let's try to select 200 points that are most "central" or just random subset?
    # Actually, for packing, it's better to have a contiguous block.
    # Let's just take the first 200.
    
    config0 = grid_full[:N].copy()
    configs_to_try.append(config0)
    
    # Config 2: Random perturbation of config0
    config1 = config0 + rng.normal(0, 0.001, size=(N, 2))
    configs_to_try.append(config1)
    
    # Config 3: Different aspect ratio? 15 rows x 14 cols?
    grid_full2 = generate_hex_grid(15, 14, r_init)
    if len(grid_full2) >= N:
        config2 = grid_full2[:N].copy()
        configs_to_try.append(config2)
        
    # Config 4: Square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_sq = np.array([(x, y) for y in ticks for x in ticks])
    config3 = grid_sq[:N].copy()
    configs_to_try.append(config3)

    # Optimize each config
    for i, config in enumerate(configs_to_try):
        if time.time() - t_start > deadline - 20:
            break
            
        config_opt = config.copy()
        N_opt = len(config_opt)
        
        # Vectorized optimization
        n_iter = 2000
        lr = 0.005
        
        for step in range(n_iter):
            if time.time() - t_start > deadline - 10:
                break
                
            diffs = config_opt[:, np.newaxis, :] - config_opt[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            min_dist = np.min(dists)
            r_curr = min_dist / 2.0
            
            b_left = config_opt[:, 0] + 0.5
            b_right = 0.5 - config_opt[:, 0]
            b_bottom = config_opt[:, 1] + 0.5
            b_top = 0.5 - config_opt[:, 1]
            
            min_b = np.min([np.min(b_left), np.min(b_right), np.min(b_bottom), np.min(b_top)])
            
            # Adaptive threshold
            threshold = 2 * r_curr * 1.2
            
            mask = dists < threshold
            # Ensure diagonal is False
            mask[~np.eye(N_opt, dtype=bool)] = mask[~np.eye(N_opt, dtype=bool)]
            
            safe_dists = np.where(dists > 1e-8, dists, 1e-8)
            
            force_mag = np.zeros_like(dists)
            force_mag[mask] = (threshold - dists[mask]) / safe_dists[mask]
            
            forces = np.sum(force_mag[:, :, np.newaxis] * diffs, axis=1)
            
            b_forces = np.zeros_like(config_opt)
            # Boundary forces
            # Use a similar soft constraint
            b_thresh = r_curr * 1.2
            
            mask_l = b_left < b_thresh
            b_forces[mask_l, 0] += (b_thresh - b_left[mask_l])
            
            mask_r = b_right < b_thresh
            b_forces[mask_r, 0] -= (b_thresh - b_right[mask_r])
            
            mask_b = b_bottom < b_thresh
            b_forces[mask_b, 1] += (b_thresh - b_bottom[mask_b])
            
            mask_t = b_top < b_thresh
            b_forces[mask_t, 1] -= (b_thresh - b_top[mask_t])
            
            config_opt += lr * (forces + b_forces)
            
            # Clip
            config_opt[:, 0] = np.clip(config_opt[:, 0], -0.5 + 1e-6, 0.5 - 1e-6)
            config_opt[:, 1] = np.clip(config_opt[:, 1], -0.5 + 1e-6, 0.5 - 1e-6)
            
        # Evaluate
        final_radius = compute_radius(config_opt)
        
        if final_radius > best_score:
            best_score = final_radius
            best_config = config_opt.copy()
            
    # Fallback
    if best_config is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_config = grid[:N]
        
    # Ensure valid
    best_config = np.clip(best_config, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_config

# EVOLVE-BLOCK-END
