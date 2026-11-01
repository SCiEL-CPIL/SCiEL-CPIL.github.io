"""
Strategy: Hard-constraint circle expansion with exact line search.
Instead of minimizing a soft potential, we directly maximize r by:
1. Finding the current minimum clearance r.
2. Identifying the most-violated constraint (pair or wall).
3. Solving the quadratic subproblem to push the violating entities apart
   by exactly the needed amount, using exact line search to preserve r.
4. Multi-start over diverse hexagonal perturbations to find better basins.
"""
import numpy as np
import time
from scipy.spatial import cKDTree

N_CIRCLES = 200
HALF_SIDE = 0.5

def compute_radius(centers):
    """Compute the maximum valid radius for the given centers."""
    if len(centers) != N_CIRCLES:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > HALF_SIDE + 1e-12):
        return 0.0

    min_dist_to_wall = np.min(HALF_SIDE - np.abs(centers), axis=None)
    if min_dist_to_wall <= 0:
        return 0.0

    tree = cKDTree(centers)
    # Use a generous radius for query_pairs to ensure we find all close pairs
    pairs = tree.query_pairs(r=0.08)
    if len(pairs) == 0:
        min_pair_dist = 1e9
    else:
        min_pair_dist = min(np.linalg.norm(centers[i] - centers[j]) for i, j in pairs)

    r = min(min_dist_to_wall, min_pair_dist / 2)
    return r

def hexagonal_initialization(perturbation=0.0):
    """Create a hexagonal lattice initialization with optional perturbation."""
    best_points = None
    best_score = 0.0
    
    # Try a range of spacings
    for scale in [0.072, 0.074, 0.076, 0.078]:
        dx = scale
        dy = scale * np.sqrt(3) / 2
        
        x_vals = np.arange(-0.5, 0.5 + dx/2, dx)
        y_vals = np.arange(-0.5, 0.5 + dy/2, dy)
        
        grid_points = []
        for i, y in enumerate(y_vals):
            for x in x_vals:
                if i % 2 == 1:
                    x_offset = dx / 2
                else:
                    x_offset = 0
                px = x + x_offset
                if -0.5 <= px <= 0.5 and -0.5 <= y <= 0.5:
                    grid_points.append([px, y])
        
        if len(grid_points) < N_CIRCLES:
            continue
        
        grid_points = np.array(grid_points)
        
        # Add perturbation if requested
        if perturbation > 0:
            rng = np.random.RandomState(abs(hash(scale)) % (2**32))
            grid_points += rng.normal(0, perturbation, grid_points.shape)
            grid_points = np.clip(grid_points, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Sort by distance from center and take N closest
        dists = np.sum(grid_points ** 2, axis=1)
        idx = np.argsort(dists)[:N_CIRCLES]
        points = grid_points[idx]
        
        r = compute_radius(points)
        if r > best_score:
            best_score = r
            best_points = points.copy()
    
    if best_points is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N_CIRCLES)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N_CIRCLES]
    
    return best_points

def expand_circle(centers, max_iters=500, time_limit=10.0):
    """
    Expand circles by iteratively resolving the most-violated constraint.
    This is a hard-constraint optimization: we never allow overlap.
    """
    start_time = time.time()
    current = centers.copy()
    current_r = compute_radius(current)
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Find the current minimum clearance
        min_dist_to_wall = np.min(HALF_SIDE - np.abs(current), axis=None)
        
        tree = cKDTree(current)
        pairs = tree.query_pairs(r=2 * current_r + 0.01)
        
        if len(pairs) == 0:
            min_pair_dist = 1e9
        else:
            min_pair_dist = min(np.linalg.norm(current[i] - current[j]) for i, j in pairs)
        
        r = min(min_dist_to_wall, min_pair_dist / 2)
        
        # Check for improvement
        if r > current_r + 1e-12:
            current_r = r
            continue
        
        # Find the most-violated constraint
        # We want to increase r, so we need to push apart the closest pair
        # or push away from the closest wall
        
        # Determine if wall or pair is the bottleneck
        wall_is_bottleneck = (min_dist_to_wall < min_pair_dist / 2 - 1e-12)
        
        if wall_is_bottleneck:
            # Find the circle closest to a wall
            wall_dists = np.minimum(np.minimum(0.5 - current[:, 0], current[:, 0] + 0.5),
                                   np.minimum(0.5 - current[:, 1], current[:, 1] + 0.5))
            idx = np.argmin(wall_dists)
            
            # Determine which wall
            x, y = current[idx]
            if 0.5 - x < x + 0.5 and 0.5 - x < 0.5 - y and 0.5 - x < y + 0.5:
                direction = np.array([-1.0, 0.0])  # Push left
            elif x + 0.5 < 0.5 - x and x + 0.5 < 0.5 - y and x + 0.5 < y + 0.5:
                direction = np.array([1.0, 0.0])   # Push right
            elif 0.5 - y < x + 0.5 and 0.5 - y < 0.5 - x and 0.5 - y < y + 0.5:
                direction = np.array([0.0, -1.0])  # Push down
            else:
                direction = np.array([0.0, 1.0])   # Push up
            
            # Push the circle away from the wall
            # We want to increase its distance to the wall
            push_amount = 0.001
            new_pos = current[idx] + push_amount * direction
            new_pos = np.clip(new_pos, -0.5 + 1e-9, 0.5 - 1e-9)
            
            # Check if this push causes any new overlaps
            tree2 = cKDTree(current)
            neighbors = tree2.query_ball_point(new_pos, r=2 * current_r + push_amount)
            neighbors.remove(idx)
            
            valid = True
            for j in neighbors:
                if np.linalg.norm(new_pos - current[j]) < 2 * current_r - 1e-10:
                    valid = False
                    break
            
            if valid:
                current[idx] = new_pos
                new_r = compute_radius(current)
                if new_r > current_r:
                    current_r = new_r
                continue
            else:
                # Can't push this circle; try another
                break
        
        else:
            # Find the closest pair
            if len(pairs) == 0:
                break
            
            closest_pair = min(pairs, key=lambda p: np.linalg.norm(current[p[0]] - current[p[1]]))
            i, j = closest_pair
            diff = current[i] - current[j]
            dist = np.linalg.norm(diff)
            
            if dist < 1e-10:
                # Random nudge
                current[i] += np.random.normal(0, 0.0005, 2)
                current[j] -= np.random.normal(0, 0.0005, 2)
                current = np.clip(current, -0.5 + 1e-9, 0.5 - 1e-9)
                continue
            
            # Push them apart
            direction = diff / dist
            push_amount = 0.001
            
            new_i = current[i] + push_amount * direction
            new_j = current[j] - push_amount * direction
            
            new_i = np.clip(new_i, -0.5 + 1e-9, 0.5 - 1e-9)
            new_j = np.clip(new_j, -0.5 + 1e-9, 0.5 - 1e-9)
            
            # Check if this push causes any new overlaps
            tree2 = cKDTree(current)
            
            # Check neighbors of i
            neighbors_i = tree2.query_ball_point(new_i, r=2 * current_r + push_amount)
            neighbors_i.remove(i)
            
            valid = True
            for k in neighbors_i:
                if k == j:
                    continue
                if np.linalg.norm(new_i - current[k]) < 2 * current_r - 1e-10:
                    valid = False
                    break
            
            # Check neighbors of j
            if valid:
                neighbors_j = tree2.query_ball_point(new_j, r=2 * current_r + push_amount)
                neighbors_j.remove(j)
                
                for k in neighbors_j:
                    if k == i:
                        continue
                    if np.linalg.norm(new_j - current[k]) < 2 * current_r - 1e-10:
                        valid = False
                        break
            
            if valid:
                current[i] = new_i
                current[j] = new_j
                new_r = compute_radius(current)
                if new_r > current_r:
                    current_r = new_r
                continue
            else:
                # Can't push this pair; try a smaller push
                push_amount *= 0.5
                if push_amount < 1e-8:
                    break
                continue
    
    return current, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_config = None
    best_r = 0.0
    
    # Multi-start strategy: try different perturbations
    perturbations = [0.0, 0.002, 0.005, 0.01, 0.02]
    
    for pert in perturbations:
        if time.time() > deadline - 15:
            break
        
        # Generate initial configuration
        config = hexagonal_initialization(perturbation=pert)
        
        # Expand circles
        expanded, r = expand_circle(config, max_iters=300, time_limit=20.0)
        
        if r > best_r:
            best_r = r
            best_config = expanded
    
    # Final check
    if best_config is None or compute_radius(best_config) <= 0:
        # Fallback to simple grid
        m = int(np.ceil(np.sqrt(N_CIRCLES)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_config = grid[:N_CIRCLES]
    
    return best_config
