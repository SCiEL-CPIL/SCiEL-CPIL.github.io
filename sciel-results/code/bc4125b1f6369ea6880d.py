import numpy as np
import time

N = 200

def compute_radius(points):
    if points is None:
        return 0.0
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (N, 2) or not np.all(np.isfinite(points)):
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist_sq = np.min(dist_sq)
    x = points[:, 0]
    y = points[:, 1]
    dist_x = 0.5 - np.abs(x)
    dist_y = 0.5 - np.abs(y)
    min_wall_dist = np.min(np.minimum(dist_x, dist_y))
    r = np.min([np.sqrt(min_pair_dist_sq)/2, min_wall_dist])
    return float(r)

def hex_grid_init(n_rows, jitter=0.0):
    """Generate hexagonal lattice with optional jitter."""
    c_base = N // n_rows
    rem = N % n_rows
    counts = [c_base] * n_rows
    for i in range(rem):
        counts[i] += 1
    
    pts = []
    y = 0.0
    for i, c in enumerate(counts):
        xs = np.arange(c) - (c - 1) / 2.0
        if i % 2 == 1:
            xs += 0.5
        for x in xs:
            pts.append([x, y])
        y += np.sqrt(3) / 2.0
    
    pts = np.array(pts, dtype=np.float64)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    max_extent = max(np.max(np.abs(pts[:, 0])), np.max(np.abs(pts[:, 1])))
    if max_extent > 0:
        scale = 0.45 / max_extent
        pts *= scale
    
    if jitter > 0:
        pts += np.random.randn(N, 2) * jitter
        pts = np.clip(pts, -0.49, 0.49)
    
    return pts

def repulsive_relax(pts, n_iters=200, deadline=None):
    """Repulsive force relaxation to spread points evenly."""
    pts = pts.copy()
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, 1e-10)
        
        # Soft repulsion: force ~ 1/d^3
        dist = np.sqrt(dist_sq)
        force = 1.0 / (dist_sq * dist)
        np.fill_diagonal(force, 0)
        
        # Net force on each point
        net_force = np.sum(force[:, :, np.newaxis] * diff, axis=1)
        
        # Wall repulsion
        x = pts[:, 0]
        y = pts[:, 1]
        margin = 0.05
        fx = np.zeros_like(x)
        fy = np.zeros_like(y)
        fx[x < -0.5 + margin] += (margin - (x[x < -0.5 + margin] + 0.5)) * 10
        fx[x > 0.5 - margin] -= (margin - (0.5 - x[x > 0.5 - margin])) * 10
        fy[y < -0.5 + margin] += (margin - (y[y < -0.5 + margin] + 0.5)) * 10
        fy[y > 0.5 - margin] -= (margin - (0.5 - y[y > 0.5 - margin])) * 10
        
        lr = 0.01 * (1.0 - it / max(n_iters, 1))
        pts += lr * (net_force + np.column_stack((fx, fy)))
        pts = np.clip(pts, -0.5, 0.5)
    
    return pts

def optimize(pts, deadline, lambda_start=20.0, lambda_end=200.0, n_steps=800, lr_start=0.002):
    """Multi-stage gradient optimization with adaptive lambda."""
    pts = pts.copy()
    best_r = compute_radius(pts)
    best_pts = pts.copy()
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Adaptive lambda: start low for exploration, increase for exploitation
        progress = step / max(n_steps - 1, 1)
        lambda_val = lambda_start + (lambda_end - lambda_start) * progress
        
        # Compute distances
        diff = pts[:, np.newaxis, :] - points[np.newaxis, :] if False else pts[:, np.newaxis, :] - pts[np.newaxis, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Pairwise forces using soft-min surrogate
        w_pairs = np.exp(-lambda_val * dist_sq)
        w_pairs[~np.isfinite(w_pairs)] = 0
        sum_w_pairs = np.sum(w_pairs, axis=1)
        
        term = w_pairs[:, :, np.newaxis] * (-2 * lambda_val) * diff
        grad_pairs = np.sum(term, axis=1)
        sum_w_pairs_safe = np.where(sum_w_pairs > 0, sum_w_pairs, 1.0)
        grad_pairs /= sum_w_pairs_safe[:, np.newaxis]
        
        # Wall forces - smooth quadratic penalty
        x = pts[:, 0]
        y = pts[:, 1]
        
        d_x_l = x + 0.5
        d_x_r = 0.5 - x
        d_y_b = y + 0.5
        d_y_t = 0.5 - y
        
        # Smooth wall gradient: push when within margin
        margin = 0.04
        wall_strength = 8.0
        
        grad_x_walls = np.zeros_like(x)
        grad_y_walls = np.zeros_like(y)
        
        mask_xl = d_x_l < margin
        mask_xr = d_x_r < margin
        mask_yb = d_y_b < margin
        mask_yt = d_y_t < margin
        
        grad_x_walls[mask_xl] -= wall_strength * (margin - d_x_l[mask_xl])
        grad_x_walls[mask_xr] += wall_strength * (margin - d_x_r[mask_xr])
        grad_y_walls[mask_yb] -= wall_strength * (margin - d_y_b[mask_yb])
        grad_y_walls[mask_yt] += wall_strength * (margin - d_y_t[mask_yt])
        
        grad_walls = np.column_stack((grad_x_walls, grad_y_walls))
        
        total_grad = grad_pairs + grad_walls
        
        # Adaptive learning rate with cosine annealing
        lr = lr_start * (0.5 * (1 + np.cos(np.pi * progress)) + 0.1)
        
        pts_new = pts - lr * total_grad
        pts_new = np.clip(pts_new, -0.5, 0.5)
        
        # Occasionally add small noise to escape local minima
        if step % 100 == 50:
            pts_new += np.random.randn(N, 2) * 0.003
            pts_new = np.clip(pts_new, -0.5, 0.5)
        
        pts = pts_new
        
        # Check improvement periodically
        if step % 30 == 0:
            r = compute_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    
    return best_pts, best_r

def polish(pts, deadline):
    """High-precision local optimization focusing on exact constraints."""
    pts = pts.copy()
    best_r = compute_radius(pts)
    best_pts = pts.copy()
    
    n_steps = 1500
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Use a sharper lambda for exploitation
        lambda_val = 500.0
        
        w_pairs = np.exp(-lambda_val * dist_sq)
        w_pairs[~np.isfinite(w_pairs)] = 0
        sum_w_pairs = np.sum(w_pairs, axis=1)
        
        term = w_pairs[:, :, np.newaxis] * (-2 * lambda_val) * diff
        grad_pairs = np.sum(term, axis=1)
        sum_w_pairs_safe = np.where(sum_w_pairs > 0, sum_w_pairs, 1.0)
        grad_pairs /= sum_w_pairs_safe[:, np.newaxis]
        
        # Tight wall constraints
        x = pts[:, 0]
        y = pts[:, 1]
        margin = 0.02
        wall_strength = 20.0
        
        d_x_l = x + 0.5
        d_x_r = 0.5 - x
        d_y_b = y + 0.5
        d_y_t = 0.5 - y
        
        grad_x_walls = np.zeros_like(x)
        grad_y_walls = np.zeros_like(y)
        
        mask_xl = d_x_l < margin
        mask_xr = d_x_r < margin
        mask_yb = d_y_b < margin
        mask_yt = d_y_t < margin
        
        grad_x_walls[mask_xl] -= wall_strength * (margin - d_x_l[mask_xl])
        grad_x_walls[mask_xr] += wall_strength * (margin - d_x_r[mask_xr])
        grad_y_walls[mask_yb] -= wall_strength * (margin - d_y_b[mask_yb])
        grad_y_walls[mask_yt] += wall_strength * (margin - d_y_t[mask_yt])
        
        total_grad = grad_pairs + np.column_stack((grad_x_walls, grad_y_walls))
        
        # Very small learning rate for precision
        lr = 0.0001 * (1.0 - step / max(n_steps, 1))
        
        pts_new = pts - lr * total_grad
        pts_new = np.clip(pts_new, -0.5, 0.5)
        
        pts = pts_new
        
        if step % 50 == 0:
            r = compute_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    best_r = 0
    best_points = None
    
    # Generate diverse initializations
    inits = []
    
    # Hexagonal grids with various row counts
    for n_rows in [12, 13, 14, 15, 16]:
        inits.append(hex_grid_init(n_rows, jitter=0.0))
        inits.append(hex_grid_init(n_rows, jitter=0.01))
    
    # Random initializations
    for _ in range(5):
        pts = np.random.rand(N, 2) - 0.5
        inits.append(pts)
    
    # Optimize each initialization with repulsive relaxation first
    for init in inits:
        if time.time() > deadline:
            break
        
        # Repulsive relaxation to get a better starting point
        relaxed = repulsive_relax(init, n_iters=100, deadline=deadline)
        
        pts, r = optimize(relaxed, deadline)
        
        if r > best_r:
            best_r = r
            best_points = pts.copy()
    
    # Final polish on best solution
    if best_points is not None and time.time() < deadline - 3:
        polish_deadline = min(deadline, time.time() + 8)
        best_points, best_r = polish(best_points, polish_deadline)
    
    if best_points is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
    
    return best_points
