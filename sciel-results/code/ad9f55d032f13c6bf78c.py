"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    t_start = time.time()
    deadline = 145.0  # seconds, with safety margin before 150s kill
    
    def compute_radius(pts):
        """Compute the maximum radius for given centers."""
        # Pairwise distances
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Boundary distances
        b_left = pts[:, 0] + 0.5
        b_right = 0.5 - pts[:, 0]
        b_bottom = pts[:, 1] + 0.5
        b_top = 0.5 - pts[:, 1]
        min_bound = np.min([np.min(b_left), np.min(b_right), np.min(b_bottom), np.min(b_top)])
        
        return min(min_pair / 2.0, min_bound)
    
    def make_hex_grid(n_rows, n_cols, spacing):
        """Create a hexagonal grid of points."""
        points = []
        dx = spacing
        dy = spacing * np.sqrt(3)
        
        for row in range(n_rows):
            y = -0.5 + spacing/2 + row * dy
            offset = 0 if row % 2 == 0 else dx / 2
            for col in range(n_cols):
                x = -0.5 + spacing/2 + offset + col * dx
                points.append([x, y])
        
        points = np.array(points)
        # Center the points
        points[:, 0] -= np.mean(points[:, 0])
        points[:, 1] -= np.mean(points[:, 1])
        return points
    
    def make_square_grid(n_rows, n_cols, spacing):
        """Create a square grid of points."""
        points = []
        for row in range(n_rows):
            for col in range(n_cols):
                x = -0.5 + spacing/2 + col * spacing
                y = -0.5 + spacing/2 + row * spacing
                points.append([x, y])
        
        points = np.array(points)
        points[:, 0] -= np.mean(points[:, 0])
        points[:, 1] -= np.mean(points[:, 1])
        return points
    
    def local_search(pts, max_iter=5000, lr_init=0.002, lr_decay=0.999):
        """Local search to optimize packing."""
        pts = pts.copy()
        lr = lr_init
        best_local = compute_radius(pts)
        best_pts = pts.copy()
        
        for it in range(max_iter):
            if time.time() - t_start > deadline - 5:
                break
            
            # Compute pairwise distances
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Current radius
            min_pair = np.min(dists)
            b_left = pts[:, 0] + 0.5
            b_right = 0.5 - pts[:, 0]
            b_bottom = pts[:, 1] + 0.5
            b_top = 0.5 - pts[:, 1]
            min_bound = np.min([np.min(b_left), np.min(b_right), np.min(b_bottom), np.min(b_top)])
            r_curr = min(min_pair / 2.0, min_bound)
            
            # Forces from close pairs
            threshold = 2 * r_curr * 1.5
            mask = dists < threshold
            np.fill_diagonal(mask, False)
            
            safe_dists = np.where(dists > 1e-10, dists, 1e-10)
            force_mag = np.zeros_like(dists)
            force_mag[mask] = (threshold - dists[mask]) / safe_dists[mask]
            
            forces = np.sum(force_mag[:, :, np.newaxis] * diffs, axis=1)
            
            # Boundary forces
            b_forces = np.zeros_like(pts)
            b_thresh = r_curr * 1.5
            
            mask_l = b_left < b_thresh
            b_forces[mask_l, 0] += (b_thresh - b_left[mask_l])
            
            mask_r = b_right < b_thresh
            b_forces[mask_r, 0] -= (b_thresh - b_right[mask_r])
            
            mask_b = b_bottom < b_thresh
            b_forces[mask_b, 1] += (b_thresh - b_bottom[mask_b])
            
            mask_t = b_top < b_thresh
            b_forces[mask_t, 1] -= (b_thresh - b_top[mask_t])
            
            # Update
            pts += lr * (forces + b_forces)
            pts[:, 0] = np.clip(pts[:, 0], -0.5 + 1e-8, 0.5 - 1e-8)
            pts[:, 1] = np.clip(pts[:, 1], -0.5 + 1e-8, 0.5 - 1e-8)
            
            # Decay learning rate
            lr *= lr_decay
            
            # Track best
            r = compute_radius(pts)
            if r > best_local:
                best_local = r
                best_pts = pts.copy()
        
        return best_pts, best_local
    
    def scipy_optimize(pts, max_iter=1000):
        """Use scipy to optimize the packing."""
        def objective(x):
            pts = x.reshape(-1, 2)
            # Penalize overlaps and boundary violations
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # We want to maximize min distance, so minimize negative
            min_pair = np.min(dists)
            
            b_left = pts[:, 0] + 0.5
            b_right = 0.5 - pts[:, 0]
            b_bottom = pts[:, 1] + 0.5
            b_top = 0.5 - pts[:, 1]
            min_bound = np.min([np.min(b_left), np.min(b_right), np.min(b_bottom), np.min(b_top)])
            
            r = min(min_pair / 2.0, min_bound)
            
            # Penalty for being outside bounds
            penalty = 0
            penalty += np.sum(np.maximum(0, np.abs(pts[:, 0]) - 0.5 + 1e-6)) * 100
            penalty += np.sum(np.maximum(0, np.abs(pts[:, 1]) - 0.5 + 1e-6)) * 100
            
            return -r + penalty
        
        x0 = pts.flatten()
        res = minimize(objective, x0, method='L-BFGS-B', maxiter=max_iter,
                      bounds=[(-0.5, 0.5)] * (2 * N))
        
        if res.success:
            pts_opt = res.x.reshape(-1, 2)
            r = compute_radius(pts_opt)
            return pts_opt, r
        
        return pts, compute_radius(pts)
    
    best_config = None
    best_score = -1
    
    rng = np.random.RandomState(42)
    
    # Generate multiple initial configurations
    configs = []
    
    # Hexagonal grids with different spacings
    for n_rows, n_cols in [(13, 16), (14, 15), (15, 14), (16, 13)]:
        if n_rows * n_cols >= N:
            spacing = 1.0 / max(n_rows, n_cols) * 0.95
            grid = make_hex_grid(n_rows, n_cols, spacing)
            if len(grid) >= N:
                configs.append(grid[:N].copy())
    
    # Square grids
    for m in [14, 15, 16]:
        spacing = 1.0 / m * 0.95
        grid = make_square_grid(m, m, spacing)
        if len(grid) >= N:
            configs.append(grid[:N].copy())
    
    # Random configurations
    for _ in range(3):
        pts = rng.uniform(-0.45, 0.45, size=(N, 2))
        configs.append(pts)
    
    # Optimize each configuration
    for i, config in enumerate(configs):
        if time.time() - t_start > deadline - 20:
            break
        
        # First pass: local search
        pts_ls, score_ls = local_search(config, max_iter=3000, lr_init=0.003, lr_decay=0.9995)
        
        if score_ls > best_score:
            best_score = score_ls
            best_config = pts_ls.copy()
        
        # Second pass: scipy optimization (if time permits)
        if time.time() - t_start < deadline - 30:
            pts_scipy, score_scipy = scipy_optimize(pts_ls, max_iter=500)
            
            if score_scipy > best_score:
                best_score = score_scipy
                best_config = pts_scipy.copy()
        
        # Third pass: fine-tuning local search
        if time.time() - t_start < deadline - 10:
            pts_fine, score_fine = local_search(pts_scipy, max_iter=2000, lr_init=0.001, lr_decay=0.9999)
            
            if score_fine > best_score:
                best_score = score_fine
                best_config = pts_fine.copy()
    
    # Fallback
    if best_config is None:
        m = 15
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_config = grid[:N]
    
    # Ensure valid
    best_config = np.clip(best_config, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_config

# EVOLVE-BLOCK-END
