"""Binary search on radius with projected gradient descent for 50 circles in unit square."""
import numpy as np
import time

N = 50
HALF = 0.5


def _score(c):
    """Largest common radius for centres c (N,2)."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    min_bound = np.min(HALF - np.abs(c))
    return min(min_pair, min_bound)


def _feasible(pts, r, max_iter=2000, deadline=None):
    """
    Try to find a configuration of N circles of radius r inside the square.
    Uses projected gradient descent to push overlapping circles apart and
    keep them within bounds. Returns (success, pts).
    """
    pts = pts.astype(np.float64).copy()
    n = len(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute all pairwise distances and forces
        diff = pts[:, None, :] - pts[None, :, :]  # (n, n, 2)
        dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
        np.fill_diagonal(dist, np.inf)
        
        # Compute forces for overlapping pairs
        overlap = (dist < 2 * r) & (dist > 1e-12)
        force_mag = np.where(overlap, (2 * r - dist), 0.0)
        inv_dist = np.where(dist > 1e-12, 1.0 / dist, 0.0)
        force = diff * (force_mag * inv_dist)[:, :, None]
        grad = np.sum(force, axis=1)  # (n, 2)
        
        # Wall constraints: push away from walls
        wall_force = np.zeros_like(pts)
        for k in range(2):
            too_right = pts[:, k] > HALF - r
            too_left = pts[:, k] < -HALF + r
            wall_force[too_right, k] = -(pts[too_right, k] - (HALF - r))
            wall_force[too_left, k] = (pts[too_left, k] - (-HALF + r))
        
        grad += wall_force
        
        # Adaptive step size
        step = 0.05
        new_pts = pts + step * grad
        new_pts = np.clip(new_pts, -HALF + r, HALF - r)
        
        # Check if all constraints are satisfied
        min_pair = np.min(dist)
        min_wall = np.min(HALF - np.abs(new_pts))
        
        if min_pair >= 2 * r - 1e-10 and min_wall >= r - 1e-10:
            return True, new_pts
        
        pts = new_pts
        
        # Early termination if not making progress
        if it > 100:
            cur_min = min(np.min(np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))),
                          np.min(HALF - np.abs(pts)))
            if cur_min < 2 * r - 1e-8:
                if it > 500:
                    break
    
    # Final check
    dist = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    min_wall = np.min(HALF - np.abs(pts))
    success = (min_pair >= 2 * r - 1e-10) and (min_wall >= r - 1e-10)
    return success, pts


def _hex_seed(r):
    """Generate a hexagonal lattice seed with spacing 2r."""
    dx = 2.0 * r
    dy = np.sqrt(3.0) * r
    xs, ys = [], []
    row = 0
    y = -HALF + r
    while y <= HALF - r + 1e-9:
        x = -HALF + r + (0.5 * dx if (row % 2) else 0.0)
        while x <= HALF - r + 1e-9:
            xs.append(x)
            ys.append(y)
            x += dx
        y += dy
        row += 1
    pts = np.array(list(zip(xs, ys)))
    if len(pts) >= N:
        d2 = np.sum(pts * pts, axis=1)
        idx = np.argsort(d2)[:N]
        pts = pts[idx]
    else:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, yy) for yy in ticks for x in ticks])
        pts = np.vstack([pts, grid[:N - len(pts)]])
    return np.clip(pts, -HALF + r, HALF - r)


def _grid_seed(r):
    """Generate a grid seed."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        d2 = np.sum(grid * grid, axis=1)
        idx = np.argsort(d2)[:N]
        grid = grid[idx]
    return np.clip(grid, -HALF + r, HALF - r)


def _soft_min_score(c, beta=50.0):
    """Smooth approximation of the minimum radius using log-sum-exp."""
    c = np.asarray(c, dtype=np.float64)
    d = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    # Pairwise distances / 2
    pair_scores = 0.5 * np.sqrt(dist_sq)
    # Wall distances
    wall_scores = HALF - np.abs(c)
    
    # Soft min of pair scores
    pair_min = -np.log(np.sum(np.exp(-beta * pair_scores))) / beta
    # Soft min of wall scores  
    wall_min = -np.log(np.sum(np.exp(-beta * wall_scores))) / beta
    
    # Soft min of both
    all_scores = np.concatenate([pair_min, wall_min])
    soft_min = -np.log(np.sum(np.exp(-beta * all_scores))) / beta
    return soft_min


def _polish(pts, max_iter=5000, deadline=None):
    """
    High-precision polish using smooth surrogate and adaptive steps.
    Maximizes the minimum distance (radius) using soft-min surrogate.
    """
    pts = pts.astype(np.float64).copy()
    n = len(pts)
    best_score = _score(pts)
    best_pts = pts.copy()
    
    beta = 30.0  # Start with moderate sharpness
    step = 0.01
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient of soft-min score
        d = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Weights for soft-min: w_ij = exp(-beta * dist_ij/2) / sum(exp(...))
        pair_scores = 0.5 * dist
        exp_pair = np.exp(-beta * pair_scores)
        exp_pair[np.diag_indices_from(exp_pair)] = 0
        pair_weights = exp_pair / np.sum(exp_pair)
        
        # Gradient contribution from pairs: for each i, sum over j of w_ij * (pts_i - pts_j) / dist_ij
        # This is the gradient of -soft_min (we want to maximize soft_min)
        inv_dist = np.where(dist > 1e-12, 1.0 / dist, 0.0)
        pair_grad = np.sum(pair_weights[:, :, None] * d * inv_dist[:, :, None], axis=1)
        
        # Wall gradient
        wall_scores = HALF - np.abs(pts)
        exp_wall = np.exp(-beta * wall_scores)
        wall_weights = exp_wall / np.sum(exp_wall)
        
        # Gradient of wall score w.r.t. pts
        wall_grad = np.zeros_like(pts)
        for k in range(2):
            # d(HALF - |x|)/dx = -sign(x)
            sign_x = np.sign(pts[:, k])
            sign_x[sign_x == 0] = 1.0
            wall_grad[:, k] = -sign_x * wall_weights[:, k]
        
        # Total gradient (we want to maximize, so move in +grad direction)
        grad = pair_grad + wall_grad
        
        # Normalize gradient for stable steps
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad = grad / grad_norm
        
        # Adaptive step size
        new_pts = pts + step * grad
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_score = _score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts
            # Increase step slightly on success
            step = min(step * 1.1, 0.05)
        else:
            # Decrease step on failure
            step = step * 0.7
        
        # Increase beta for sharper approximation over time
        if it % 100 == 0 and it > 0:
            beta = min(beta * 1.5, 200.0)
        
        # Convergence check
        if it > 500 and new_score - best_score < 1e-12:
            if it > 1000:
                break
    
    return best_pts, best_score


def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    # Start with a reasonable radius estimate
    r_low = 0.06
    r_high = 0.08
    
    best_pts = _hex_seed(0.071).copy()
    best_r = _score(best_pts)
    
    # Binary search on radius
    for it in range(30):
        if time.time() > deadline:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        success = False
        trial_pts = None
        
        seeds = [_hex_seed(r_mid), _grid_seed(r_mid)]
        for _ in range(2):
            seed = _hex_seed(r_mid) + np.random.randn(N, 2) * 0.01
            seed = np.clip(seed, -HALF + r_mid, HALF + r_mid)
            seeds.append(seed)
        
        for seed in seeds:
            if time.time() > deadline:
                break
            ok, pts = _feasible(seed, r_mid, max_iter=1000, deadline=deadline)
            if ok:
                success = True
                trial_pts = pts
                break
        
        if success:
            r_low = r_mid
            best_pts = trial_pts.copy()
            best_r = _score(best_pts)
        else:
            r_high = r_mid
    
    # Final refinement with smooth polish
    if time.time() < deadline - 5:
        polished_pts, polished_r = _polish(best_pts, max_iter=3000, deadline=deadline - 2)
        if polished_r > best_r:
            best_pts = polished_pts
            best_r = polished_r
    
    return best_pts
