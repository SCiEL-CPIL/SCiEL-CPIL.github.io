import numpy as np
import time

N = 200

def compute_radius(points):
    """Compute the radius for a set of circle centers in [-0.5, 0.5]^2."""
    if points is None:
        return 0.0
    points = np.asarray(points)
    if points.shape != (N, 2) or not np.all(np.isfinite(points)):
        return 0.0
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    # Wall distances
    x = points[:, 0]
    y = points[:, 1]
    min_wall = np.min(np.minimum(np.minimum(0.5 - np.abs(x), 0.5 - np.abs(y)), 1e15))
    # If any point is outside, wall distance is negative -> invalid
    if np.any(np.abs(x) > 0.5 + 1e-12) or np.any(np.abs(y) > 0.5 + 1e-12):
        return 0.0
    return min(min_pair_dist / 2.0, min_wall)

def hex_init(n_cols=None):
    """Generate a hexagonal lattice of N points, scaled to fit in [-0.5, 0.5]^2."""
    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    
    points = []
    row = 0
    while len(points) < N:
        n_in_row = n_cols if row % 2 == 0 else n_cols
        # Alternate rows offset
        offset = 0.0 if row % 2 == 0 else 0.5
        for col in range(n_in_row):
            if len(points) >= N:
                break
            x = col + offset
            y = row * np.sqrt(3) / 2.0
            points.append([x, y])
        row += 1
    
    pts = np.array(points[:N])
    
    # Center and scale
    cx = (pts[:, 0].min() + pts[:, 0].max()) / 2.0
    cy = (pts[:, 1].min() + pts[:, 1].max()) / 2.0
    pts -= [cx, cy]
    
    max_extent = max(np.max(np.abs(pts[:, 0])), np.max(np.abs(pts[:, 1])))
    scale = 0.49 / max_extent if max_extent > 0 else 1.0
    pts *= scale
    
    return pts

def repulsive_relax(pts, n_iter=500, dt=0.01):
    """Apply repulsive force-based relaxation to push circles apart."""
    pts = pts.copy()
    
    for it in range(n_iter):
        # Compute pairwise forces
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        
        # Force: repulsive, stronger when close
        # F = diff / dist^3 (Coulomb-like)
        # But we want to maximize minimum distance, so use inverse distance
        force = np.zeros_like(pts)
        mask = dist_sq > 0
        # Normalize by distance cubed for inverse-square force
        inv_dist3 = np.where(mask, 1.0 / (dist_sq * dist + 1e-12), 0.0)
        force = np.sum(diff * inv_dist3[:, :, np.newaxis], axis=1)
        
        # Wall forces: push away from walls
        x = pts[:, 0]
        y = pts[:, 1]
        wall_force_x = np.zeros(N)
        wall_force_y = np.zeros(N)
        # Left wall
        dl = x + 0.5
        wall_force_x += np.where(dl < 0.1, 1.0 / (dl + 1e-6), 0.0)
        # Right wall
        dr = 0.5 - x
        wall_force_x -= np.where(dr < 0.1, 1.0 / (dr + 1e-6), 0.0)
        # Bottom wall
        db = y + 0.5
        wall_force_y += np.where(db < 0.1, 1.0 / (db + 1e-6), 0.0)
        # Top wall
        dt_ = 0.5 - y
        wall_force_y -= np.where(dt_ < 0.1, 1.0 / (dt_ + 1e-6), 0.0)
        
        force[:, 0] += wall_force_x
        force[:, 1] += wall_force_y
        
        # Update
        pts += dt * force
        pts = np.clip(pts, -0.49, 0.49)
        
        # Decay step size
        dt *= 0.995
    
    return pts

def soft_min_radius(pts, lam=50.0):
    """Compute a soft approximation of the minimum radius using log-sum-exp."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Soft min of pairwise distances (we want to maximize the min)
    # Use -log-sum-exp(-lam * d^2) as proxy for min d^2
    # Actually, let's work with the objective: maximize min(d_ij/2, wall_i)
    # We minimize F = log(sum exp(-lam * d_ij^2)) + log(sum exp(-lam * wall_i^2))
    
    w_pairs = np.exp(-lam * dist_sq)
    w_pairs[~np.isfinite(w_pairs)] = 0
    F_pairs = np.log(np.sum(w_pairs) + 1e-300)
    
    # Wall terms
    x = pts[:, 0]
    y = pts[:, 1]
    d_x_l = x + 0.5
    d_x_r = 0.5 - x
    d_y_b = y + 0.5
    d_y_t = 0.5 - y
    
    w_walls = (np.exp(-lam * d_x_l**2) + np.exp(-lam * d_x_r**2) + 
               np.exp(-lam * d_y_b**2) + np.exp(-lam * d_y_t**2))
    F_walls = np.log(np.sum(w_walls) + 1e-300)
    
    return F_pairs + F_walls

def polish(pts, lam=100.0, n_iter=2000, lr_init=0.005):
    """Polish the packing using gradient descent on the soft-min objective."""
    pts = pts.copy()
    best_r = compute_radius(pts)
    best_pts = pts.copy()
    
    for it in range(n_iter):
        # Compute distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Weights for pairs
        w_pairs = np.exp(-lam * dist_sq)
        w_pairs[~np.isfinite(w_pairs)] = 0
        
        sum_w_pairs = np.sum(w_pairs, axis=1)
        
        # Gradient from pairs
        term = w_pairs[:, :, np.newaxis] * (-2 * lam) * diff
        grad_pairs = np.sum(term, axis=1)
        sum_w_pairs_safe = np.where(sum_w_pairs > 0, sum_w_pairs, 1.0)
        grad_pairs /= sum_w_pairs_safe[:, np.newaxis]
        
        # Wall terms
        x = pts[:, 0]
        y = pts[:, 1]
        
        d_x_l = x + 0.5
        d_x_r = 0.5 - x
        d_y_b = y + 0.5
        d_y_t = 0.5 - y
        
        w_x_l = np.exp(-lam * d_x_l**2)
        w_x_r = np.exp(-lam * d_x_r**2)
        w_y_b = np.exp(-lam * d_y_b**2)
        w_y_t = np.exp(-lam * d_y_t**2)
        
        sum_w_walls = w_x_l + w_x_r + w_y_b + w_y_t
        
        grad_x_walls = (w_x_l * (-2 * lam * d_x_l) + w_x_r * (2 * lam * d_x_r)) / sum_w_walls
        grad_y_walls = (w_y_b * (-2 * lam * d_y_b) + w_y_t * (2 * lam * d_y_t)) / sum_w_walls
        
        grad_walls = np.column_stack((grad_x_walls, grad_y_walls))
        
        total_grad = grad_pairs + grad_walls
        
        # Adaptive learning rate
        lr = lr_init / (1 + it * 0.001)
        
        pts_new = pts - lr * total_grad
        pts_new = np.clip(pts_new, -0.499, 0.499)
        
        # Check improvement periodically
        if it % 50 == 0:
            r = compute_radius(pts_new)
            if r > best_r:
                best_r = r
                best_pts = pts_new.copy()
        
        pts = pts_new
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill
    
    best_r = 0
    best_points = None
    
    def update_best(pts):
        nonlocal best_r, best_points
        r = compute_radius(pts)
        if r > best_r:
            best_r = r
            best_points = pts.copy()
        return r
    
    # Generate multiple hexagonal initializations with different column counts
    inits = []
    for n_cols in range(10, 25):
        if time.time() > deadline:
            break
        pts = hex_init(n_cols)
        inits.append(pts)
    
    # Also add a few random initializations
    for _ in range(10):
        if time.time() > deadline:
            break
        pts = np.random.rand(N, 2) - 0.5
        inits.append(pts)
    
    # Process each initialization
    for init in inits:
        if time.time() > deadline:
            break
        
        # Step 1: Repulsive relaxation
        pts = repulsive_relax(init, n_iter=300, dt=0.005)
        update_best(pts)
        
        if time.time() > deadline:
            break
        
        # Step 2: Polish with gradient descent
        pts = polish(pts, lam=80.0, n_iter=1500, lr_init=0.003)
        update_best(pts)
        
        # Step 3: Final fine polish with higher lambda
        if time.time() < deadline - 5:
            pts = polish(pts, lam=200.0, n_iter=1000, lr_init=0.001)
            update_best(pts)
    
    # If we have time, do a few more random restarts
    while time.time() < deadline - 10:
        pts = np.random.rand(N, 2) - 0.5
        pts = repulsive_relax(pts, n_iter=200, dt=0.005)
        update_best(pts)
        if time.time() < deadline - 5:
            pts = polish(pts, lam=100.0, n_iter=1000, lr_init=0.002)
            update_best(pts)
    
    if best_points is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
    
    return best_points
