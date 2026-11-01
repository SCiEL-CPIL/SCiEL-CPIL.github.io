"""
Combined strategy: Binary search on radius with Projected Gradient Descent (PGD) feasibility solver.
"""
import numpy as np
import time

N = 50
HALF = 0.5

# EVOLVE-BLOCK-START

def _score(c):
    """
    Compute the largest common radius for centres c (N,2).
    Returns 0.0 if invalid.
    """
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    
    # Pairwise distances
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    # Wall distances
    min_bound = np.min(HALF - np.abs(c))
    
    return min(min_pair, min_bound)


def _hex_seed(r):
    """
    Generate a hexagonal lattice seed with spacing approx 2r.
    This is a good initial guess for circle packing.
    """
    dx = 2.0 * r
    dy = np.sqrt(3.0) * r
    xs, ys = [], []
    row = 0
    y = -HALF + r
    # Add a small epsilon to ensure we don't miss points due to floating point
    while y <= HALF - r + 1e-9:
        # Offset every other row for hex packing
        x_offset = 0.5 * dx if (row % 2) else 0.0
        x = -HALF + r + x_offset
        while x <= HALF - r + 1e-9:
            xs.append(x)
            ys.append(y)
            x += dx
        y += dy
        row += 1
    
    pts = np.array(list(zip(xs, ys)))
    if len(pts) >= N:
        # Select N points closest to the center
        d2 = np.sum(pts * pts, axis=1)
        idx = np.argsort(d2)[:N]
        pts = pts[idx]
    else:
        # Fallback to grid if hex lattice is too sparse (should not happen for r~0.07)
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, yy) for yy in ticks for x in ticks])
        pts = np.vstack([pts, grid[:N - len(pts)]])
        
    return np.clip(pts, -HALF + r, HALF - r)


def _grid_seed(r):
    """
    Generate a simple grid seed.
    """
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        d2 = np.sum(grid * grid, axis=1)
        idx = np.argsort(d2)[:N]
        grid = grid[idx]
    return np.clip(grid, -HALF + r, HALF - r)


def _feasible(pts, r, max_iter=500, deadline=None):
    """
    Try to find a feasible configuration for radius r using Projected Gradient Descent.
    
    Args:
        pts: Initial configuration (N, 2)
        r: Target radius
        max_iter: Maximum iterations
        deadline: Absolute time to stop
        
    Returns:
        (success, pts): success is True if all constraints are satisfied.
    """
    pts = pts.astype(np.float64).copy()
    n = len(pts)
    
    # If r is too large, even the center might not fit, but we assume r < 0.5
    # The valid region for centers is [-0.5+r, 0.5-r]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # 1. Compute forces
        diff = pts[:, None, :] - pts[None, :, :]  # (n, n, 2)
        dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
        np.fill_diagonal(dist, np.inf)
        
        # Repulsive force for overlapping pairs (dist < 2r)
        # We want to push them apart.
        # Force magnitude proportional to overlap depth
        overlap = (dist < 2 * r) & (dist > 1e-12)
        force_mag = np.where(overlap, (2 * r - dist), 0.0)
        
        # Normalize direction
        inv_dist = np.where(dist > 1e-12, 1.0 / dist, 0.0)
        # Force vector: direction * magnitude
        # diff[i,j] is vector from j to i. So force on i is in direction of diff[i,j]
        force = diff * (force_mag * inv_dist)[:, :, None]
        grad = np.sum(force, axis=1)  # Sum forces on each point
        
        # 2. Wall forces
        # If a point is too close to a wall (distance < r), push it back.
        # Actually, we will project, but gradient helps convergence.
        # Let's add a soft penalty for being close to the boundary of the valid box.
        # Valid box: [-0.5+r, 0.5-r]
        margin = r
        low_bound = -HALF + margin
        high_bound = HALF - margin
        
        wall_grad = np.zeros_like(pts)
        # Left wall
        too_left = pts[:, 0] < low_bound
        if np.any(too_left):
            wall_grad[too_left, 0] += (low_bound - pts[too_left, 0])
        # Right wall
        too_right = pts[:, 0] > high_bound
        if np.any(too_right):
            wall_grad[too_right, 0] -= (pts[too_right, 0] - high_bound)
        # Bottom wall
        too_low = pts[:, 1] < low_bound
        if np.any(too_low):
            wall_grad[too_low, 1] += (low_bound - pts[too_low, 1])
        # Top wall
        too_high = pts[:, 1] > high_bound
        if np.any(too_high):
            wall_grad[too_high, 1] -= (pts[too_high, 1] - high_bound)
            
        grad += wall_grad
        
        # 3. Update
        # Adaptive step size? Fixed step is usually fine with projection.
        step = 0.1 
        new_pts = pts + step * grad
        
        # 4. Project to valid region
        new_pts[:, 0] = np.clip(new_pts[:, 0], low_bound, high_bound)
        new_pts[:, 1] = np.clip(new_pts[:, 1], low_bound, high_bound)
        
        # 5. Check convergence / feasibility
        # Recalculate distances for new points
        diff_new = new_pts[:, None, :] - new_pts[None, :, :]
        dist_new = np.sqrt(np.sum(diff_new * diff_new, axis=2))
        np.fill_diagonal(dist_new, np.inf)
        
        min_pair = np.min(dist_new)
        min_wall = np.min(HALF - np.abs(new_pts))
        
        # Check if feasible: min_pair >= 2r and min_wall >= r
        if min_pair >= 2 * r - 1e-9 and min_wall >= r - 1e-9:
            return True, new_pts
            
        pts = new_pts
        
    # Final check
    diff_final = pts[:, None, :] - pts[None, :, :]
    dist_final = np.sqrt(np.sum(diff_final * diff_final, axis=2))
    np.fill_diagonal(dist_final, np.inf)
    min_pair = np.min(dist_final)
    min_wall = np.min(HALF - np.abs(pts))
    
    success = (min_pair >= 2 * r - 1e-9) and (min_wall >= r - 1e-9)
    return success, pts


def construct():
    """
    Main entry point.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial bounds for radius
    # Best known is ~0.071377. Let's start a bit lower to ensure feasibility.
    r_low = 0.065
    r_high = 0.075
    
    best_pts = None
    best_r = 0.0
    
    # Get initial best from r_low
    init_pts = _hex_seed(r_low)
    ok, init_pts = _feasible(init_pts, r_low, max_iter=200, deadline=deadline)
    if ok:
        best_pts = init_pts
        best_r = _score(best_pts)
        r_low = best_r # Update lower bound to actual score
    
    # Binary search
    # We will do a fixed number of iterations or until time runs out.
    # Each iteration involves multiple feasibility checks.
    
    for _ in range(25):
        if time.time() > deadline - 10:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible configuration for r_mid
        found = False
        candidate_pts = None
        
        # Use multiple seeds to escape local minima
        seeds = [
            _hex_seed(r_mid),
            _grid_seed(r_mid),
            _hex_seed(r_mid) + np.random.randn(N, 2) * 0.005,
            _grid_seed(r_mid) + np.random.randn(N, 2) * 0.005
        ]
        
        for seed in seeds:
            if time.time() > deadline - 10:
                break
                
            # Clip seed to valid region for r_mid
            seed = np.clip(seed, -HALF + r_mid, HALF - r_mid)
            
            ok, pts = _feasible(seed, r_mid, max_iter=300, deadline=deadline)
            if ok:
                found = True
                candidate_pts = pts
                break
        
        if found:
            r_low = r_mid
            best_pts = candidate_pts
            best_r = _score(best_pts)
            # If score is better than r_mid, update r_low to score
            if best_r > r_low:
                r_low = best_r
        else:
            r_high = r_mid
            
        # Early stopping if bounds are close
        if r_high - r_low < 1e-5:
            break
            
    # Final Polish: Try to improve the best solution found
    if best_pts is not None and time.time() < deadline - 5:
        # Try to push the radius a bit higher
        current_r = _score(best_pts)
        # Attempt a few local optimizations with slightly higher radius
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            target_r = current_r + 0.0005
            # Use current best as seed
            ok, polished = _feasible(best_pts, target_r, max_iter=200, deadline=deadline)
            if ok:
                new_r = _score(polished)
                if new_r > current_r:
                    best_pts = polished
                    current_r = new_r
                else:
                    break
            else:
                break

    if best_pts is None:
        # Fallback to a simple grid if something went wrong
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N]
        
    return best_pts

# EVOLVE-BLOCK-END
