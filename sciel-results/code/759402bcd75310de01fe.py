"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius_grid(points, grid):
    # points: (N,3) unit vectors, grid: (M,3) unit vectors
    # returns max over grid of min angular distance to any point
    # Use dot products: angular dist = arccos(dot)
    # For speed, compute min dot over N for each grid point, then arccos
    # dot: (M, N)
    dots = grid @ points.T  # (M, N)
    min_dot = dots.max(axis=1)  # max dot = min angle
    min_dot = np.clip(min_dot, -1.0, 1.0)
    angles = np.arccos(min_dot)
    return np.max(angles) * 180.0 / np.pi

def _build_grid(n_theta=800, n_phi=1600):
    # Fibonacci-style grid for even coverage
    i = np.arange(n_theta) + 0.5
    z = 1.0 - 2.0 * i / n_theta
    phi = 2.0 * np.pi * i / n_phi  # this is wrong, let me use a proper grid
    # Actually let's use a structured grid
    theta = np.linspace(0, np.pi, n_theta)
    phi_grid = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi_grid, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return grid

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    # Initial configuration
    pts = _fibonacci_sphere(N).astype(np.float64)
    
    # Build evaluation grid
    grid = _build_grid(800, 1600)
    n_grid = len(grid)
    
    # Current best
    best_pts = pts.copy()
    best_radius = _covering_radius_grid(pts, grid)
    
    # Gradient descent with adaptive step
    lr = 0.01
    min_lr = 1e-6
    patience = 0
    max_patience = 500
    
    while time.time() < deadline:
        # Compute gradient: for each point, push it toward the worst-covered grid regions
        # Soft approach: for each grid point, find the closest config point and accumulate gradient
        dots = grid @ pts.T  # (M, N)
        max_dot = dots.max(axis=1)  # (M,)
        max_dot_clipped = np.clip(max_dot, -1.0, 1.0)
        angles = np.arccos(max_dot_clipped)  # (M,)
        
        # Find which point is closest to each grid point
        max_idx = dots.argmax(axis=1)  # (M,)
        
        # Weight: points that are far from any config point get higher weight
        # Use a smooth weighting: weight = angle^2 or exp-based
        weights = angles ** 2  # (M,)
        
        # Gradient for each point: sum over grid points where this point is closest
        # grad_i = sum_{j: argmax_j = i} weights_j * (grid_j - pts_i) * (1/|grid_j - pts_i|)
        # But we need to project onto the tangent plane
        
        grad = np.zeros_like(pts)
        for i in range(N):
            mask = (max_idx == i)
            if not np.any(mask):
                continue
            g_pts = grid[mask]  # (k, 3)
            w = weights[mask]  # (k,)
            # Direction from pts[i] to each g_pt, normalized
            diff = g_pts - pts[i]  # (k, 3)
            norms = np.linalg.norm(diff, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            diff_normed = diff / norms  # (k, 3)
            # Weighted sum
            g = np.sum(w[:, None] * diff_normed, axis=0)  # (3,)
            # Project onto tangent plane
            g = g - np.dot(g, pts[i]) * pts[i]
            grad[i] = g
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        grad = grad / grad_norm
        
        # Take a step
        new_pts = pts + lr * grad
        
        # Normalize back to sphere
        new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        # Evaluate
        new_radius = _covering_radius_grid(new_pts, grid)
        
        if new_radius < best_radius:
            best_pts = new_pts.copy()
            best_radius = new_radius
            pts = new_pts
            patience = 0
            # Slightly increase learning rate if improving
            lr = min(lr * 1.05, 0.1)
        else:
            patience += 1
            lr *= 0.95
            if lr < min_lr:
                break
            if patience > max_patience:
                break
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # Final polish: use a finer grid for the last few iterations if time permits
    if time.time() < deadline - 10.0:
        grid_fine = _build_grid(1200, 2400)
        pts = best_pts.copy()
        best_radius = _covering_radius_grid(pts, grid_fine)
        lr = 0.005
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            dots = grid_fine @ pts.T
            max_dot = dots.max(axis=1)
            max_dot_clipped = np.clip(max_dot, -1.0, 1.0)
            angles = np.arccos(max_dot_clipped)
            max_idx = dots.argmax(axis=1)
            weights = angles ** 2
            
            grad = np.zeros_like(pts)
            for i in range(N):
                mask = (max_idx == i)
                if not np.any(mask):
                    continue
                g_pts = grid_fine[mask]
                w = weights[mask]
                diff = g_pts - pts[i]
                norms = np.linalg.norm(diff, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                diff_normed = diff / norms
                g = np.sum(w[:, None] * diff_normed, axis=0)
                g = g - np.dot(g, pts[i]) * pts[i]
                grad[i] = g
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            grad = grad / grad_norm
            
            new_pts = pts + lr * grad
            new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_radius = _covering_radius_grid(new_pts, grid_fine)
            
            if new_radius < best_radius:
                best_pts = new_pts.copy()
                best_radius = new_radius
                pts = new_pts
            else:
                lr *= 0.9
    
    return best_pts
# EVOLVE-BLOCK-END
