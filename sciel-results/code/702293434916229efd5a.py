import time
import numpy as np

N = 200

def _compute_radius(pos):
    """Compute the actual radius for a given set of positions."""
    # Pairwise distances
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    # Boundary distances
    bx = np.minimum(0.5 - pos[:, 0], pos[:, 0] + 0.5)
    by = np.minimum(0.5 - pos[:, 1], pos[:, 1] + 0.5)
    min_bnd = np.minimum(bx, by).min()
    return 0.5 * min(min_pair, min_bnd)

def _hex_init():
    """Generate a hexagonal grid of N points in the unit square."""
    # For N=200 in a 1x1 square, hexagonal spacing:
    # Area per circle ≈ 1/200 = 0.005
    # Hex packing: area per point = s^2 * sqrt(3)/2 where s = center distance
    # s^2 * sqrt(3)/2 = 0.005  =>  s = sqrt(0.01/sqrt(3)) ≈ 0.07598
    s = np.sqrt(0.01 / np.sqrt(3))
    
    points = []
    # Generate a large hex grid
    for i in range(-15, 15):
        for j in range(-15, 15):
            x = j * s + (i % 2) * s / 2
            y = i * s * np.sqrt(3) / 2
            if -0.5 < x < 0.5 and -0.5 < y < 0.5:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) >= N:
        # Take the N points closest to center to maximize boundary margin
        dist_to_center = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        idx = np.argsort(dist_to_center)[:N]
        points = points[idx]
    else:
        # Need to add more points
        needed = N - len(points)
        # Add random points, then we'll fix overlaps later
        extra = np.random.uniform(-0.45, 0.45, size=(needed, 2))
        points = np.vstack([points, extra])
    
    return points

def _square_init():
    """Square grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def _local_search(pos, max_iter=5000, time_limit=None, start_time=None):
    """
    Bottleneck elimination local search.
    Find the tightest constraint and nudge the involved circles apart.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 120
    
    pos = pos.copy()
    best_r = _compute_radius(pos)
    best_pos = pos.copy()
    
    # Precompute for efficiency
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute pairwise distances
        d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        
        # Find closest pair
        i, j = np.unravel_index(np.argmin(d), d.shape)
        min_pair_dist = d[i, j]
        
        # Boundary distances
        bx = np.minimum(0.5 - pos[:, 0], pos[:, 0] + 0.5)
        by = np.minimum(0.5 - pos[:, 1], pos[:, 1] + 0.5)
        bnd = np.minimum(bx, by)
        min_bnd_val = bnd.min()
        min_bnd_idx = np.argmin(bnd)
        
        # Current radius
        r = 0.5 * min(min_pair_dist, min_bnd_val)
        
        # Determine bottleneck
        if min_pair_dist <= min_bnd_val:
            # Pair is the bottleneck
            # Move i and j apart along their connecting vector
            diff = pos[i] - pos[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                direction = diff / dist
            else:
                direction = np.array([1.0, 0.0])
            
            # How much to move each
            # We want to increase the distance by some amount
            # Move each by half the needed increase
            # Try a small step
            step = 0.001
            pos[i] += direction * step
            pos[j] -= direction * step
        else:
            # Boundary is the bottleneck
            idx = min_bnd_idx
            # Which boundary is violated?
            x, y = pos[idx]
            dx_left = 0.5 - x   # distance to x=0.5
            dx_right = x + 0.5  # distance to x=-0.5
            dy_top = 0.5 - y    # distance to y=0.5
            dy_bot = y + 0.5    # distance to y=-0.5
            
            # Find the minimum boundary distance
            min_b = min(dx_left, dx_right, dy_top, dy_bot)
            
            if min_b == dx_left:
                direction = np.array([-1.0, 0.0])  # move left (away from x=0.5)
            elif min_b == dx_right:
                direction = np.array([1.0, 0.0])   # move right (away from x=-0.5)
            elif min_b == dy_top:
                direction = np.array([0.0, -1.0])  # move down (away from y=0.5)
            else:
                direction = np.array([0.0, 1.0])   # move up (away from y=-0.5)
            
            step = 0.001
            pos[idx] += direction * step
        
        # Clamp to square
        pos = np.clip(pos, -0.4999, 0.4999)
        
        # Check improvement
        new_r = _compute_radius(pos)
        if new_r > best_r + 1e-12:
            best_r = new_r
            best_pos = pos.copy()
        else:
            # No improvement, try a different strategy
            # Revert and try a random perturbation
            pos = best_pos.copy()
            
            # Try moving a random subset of points slightly
            n_perturb = min(10, N)
            idxs = np.random.choice(N, n_perturb, replace=False)
            pos[idxs] += np.random.normal(0, 0.0005, size=(n_perturb, 2))
            pos = np.clip(pos, -0.4999, 0.4999)
    
    return best_pos, best_r

def _local_search_v2(pos, max_iter=10000, time_limit=120, start_time=None):
    """
    Improved local search with adaptive step sizes and multiple strategies.
    """
    if start_time is None:
        start_time = time.time()
    
    pos = pos.copy()
    best_r = _compute_radius(pos)
    best_pos = pos.copy()
    
    # Step size schedule
    base_step = 0.002
    step = base_step
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute constraints
        d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        
        i, j = np.unravel_index(np.argmin(d), d.shape)
        min_pair_dist = d[i, j]
        
        bx = np.minimum(0.5 - pos[:, 0], pos[:, 0] + 0.5)
        by = np.minimum(0.5 - pos[:, 1], pos[:, 1] + 0.5)
        bnd = np.minimum(bx, by)
        min_bnd_val = bnd.min()
        min_bnd_idx = np.argmin(bnd)
        
        r = 0.5 * min(min_pair_dist, min_bnd_val)
        
        # Determine which constraint is active
        if min_pair_dist <= min_bnd_val:
            # Pair constraint
            diff = pos[i] - pos[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                direction = np.array([1.0, 0.0])
            else:
                direction = diff / dist
            
            # Move apart
            pos[i] += direction * step
            pos[j] -= direction * step
        else:
            # Boundary constraint
            idx = min_bnd_idx
            x, y = pos[idx]
            # Determine which wall
            walls = [
                (0.5 - x, np.array([-1.0, 0.0])),  # right wall
                (x + 0.5, np.array([1.0, 0.0])),   # left wall
                (0.5 - y, np.array([0.0, -1.0])),  # top wall
                (y + 0.5, np.array([0.0, 1.0])),   # bottom wall
            ]
            walls.sort(key=lambda w: w[0])
            direction = walls[0][1]
            
            pos[idx] += direction * step
        
        # Clamp
        pos = np.clip(pos, -0.4999, 0.4999)
        
        new_r = _compute_radius(pos)
        if new_r > best_r + 1e-12:
            best_r = new_r
            best_pos = pos.copy()
            step = min(step * 1.05, 0.005)  # Slightly increase step if improving
        else:
            # Revert and try perturbation
            pos = best_pos.copy()
            step = max(step * 0.7, 0.0005)
            
            # Perturb a few random points
            n_p = min(5, N)
            idxs = np.random.choice(N, n_p, replace=False)
            pos[idxs] += np.random.normal(0, 0.001, size=(n_p, 2))
            pos = np.clip(pos, -0.4999, 0.4999)
    
    return best_pos, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    best_pos = None
    best_r = -np.inf
    
    # Try multiple initializations
    inits = [
        _hex_init(),
        _square_init(),
        np.random.uniform(-0.45, 0.45, size=(N, 2)),
    ]
    
    for init_idx, init_pos in enumerate(inits):
        if time.time() > deadline - 5:
            break
        
        # For each init, do local search
        pos, r = _local_search_v2(
            init_pos,
            max_iter=20000,
            time_limit=deadline - time.time() - 3,
            start_time=start_time
        )
        
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
        
        # Also try a perturbed version of the best found
        if time.time() > deadline - 10:
            break
        
        # Add small random perturbation and search again
        perturbed = best_pos + np.random.normal(0, 0.001, size=(N, 2))
        perturbed = np.clip(perturbed, -0.4999, 0.4999)
        
        pos2, r2 = _local_search_v2(
            perturbed,
            max_iter=10000,
            time_limit=deadline - time.time() - 3,
            start_time=start_time
        )
        
        if r2 > best_r:
            best_r = r2
            best_pos = pos2.copy()
    
    # Final safety check
    if best_pos is None:
        best_pos = _square_init()
    
    # Ensure validity
    best_pos = np.clip(best_pos, -0.4999, 0.4999)
    
    return best_pos
