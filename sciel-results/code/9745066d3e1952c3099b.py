"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s margin for the 150s limit


def _compute_radius(c):
    """Compute the maximum radius for given centers."""
    # Distance to boundaries
    boundary_dist = 0.5 - np.abs(c).max(axis=1)
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = d.min()
    return min(boundary_dist.min(), min_pair_dist / 2.0)


def _initial_hex_grid(n, offset_x=0.0, offset_y=0.0, scale=1.0):
    """
    Generate a hexagonal grid of points in [-0.5, 0.5]^2.
    """
    points = []
    
    # Hexagonal packing parameters
    s = np.sqrt(2.0 / (n * np.sqrt(3.0))) * scale
    
    row_height = s * np.sqrt(3.0) / 2.0
    num_rows = int(np.ceil(1.0 / row_height)) + 6
    num_cols = int(np.ceil(1.0 / s)) + 6
    
    start_row = -num_rows // 2
    start_col = -num_cols // 2
    
    for i in range(start_row, start_row + num_rows):
        y = i * row_height + offset_y
        if y < -0.5 - s or y > 0.5 + s:
            continue
        x_offset = (s / 2.0) if i % 2 != 0 else 0.0
        for j in range(start_col, start_col + num_cols):
            x = j * s + x_offset + offset_x
            if x < -0.5 - s or x > 0.5 + s:
                continue
            cx = np.clip(x, -0.5, 0.5)
            cy = np.clip(y, -0.5, 0.5)
            points.append([cx, cy])
            
    points = np.array(points)
    
    if len(points) == 0:
        return _initial_square_grid(n)
    
    if len(points) > n:
        # Greedy farthest point sampling
        selected = [points[0]]
        for _ in range(n - 1):
            d = cdist(points, np.array(selected))
            min_d = d.min(axis=1)
            idx = np.argmax(min_d)
            selected.append(points[idx])
        points = np.array(selected)
    elif len(points) < n:
        while len(points) < n:
            points = np.vstack([points, np.random.uniform(-0.45, 0.45, size=(1, 2))])
            
    return points[:n]


def _initial_square_grid(n):
    """Generate a square grid of n points in [-0.5, 0.5]^2."""
    m = int(np.ceil(np.sqrt(n)))
    while m * m < n:
        m += 1
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _bottleneck_polish(c, time_budget=30.0):
    """
    Targeted bottleneck elimination local search.
    Moves only the closest pair or boundary-violating points.
    """
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    start = time.time()
    
    step_size = 0.002
    no_improve_count = 0
    
    while time.time() - start < time_budget:
        # Compute current radius and identify bottlenecks
        d = cdist(best_c, best_c)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = d.min()
        boundary_dist = 0.5 - np.abs(best_c).max(axis=1)
        min_boundary = boundary_dist.min()
        
        current_r = min(min_boundary, min_pair_dist / 2.0)
        
        # Find the closest pair
        min_idx = np.unravel_index(d.argmin(), d.shape)
        i, j = min_idx
        
        # Find boundary-violating points
        boundary_idx = np.argwhere(boundary_dist < current_r + 1e-6)
        
        # Move the closest pair apart
        diff = best_c[i] - best_c[j]
        dist = d[i, j]
        if dist < 1e-10:
            # Perturb randomly if coincident
            diff = np.random.normal(0, 0.01, size=2)
            dist = np.linalg.norm(diff)
        
        # Move i away from j and j away from i
        move_i = (diff / dist) * step_size
        move_j = (-diff / dist) * step_size
        
        best_c[i] += move_i
        best_c[j] += move_j
        
        # Move boundary-violating points inward
        for idx in boundary_idx:
            k = idx[0]
            # Determine which boundary is violated
            if best_c[k, 0] > 0:
                best_c[k, 0] -= step_size
            else:
                best_c[k, 0] += step_size
            if best_c[k, 1] > 0:
                best_c[k, 1] -= step_size
            else:
                best_c[k, 1] += step_size
        
        # Clip to square
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
        
        # Compute new radius
        new_r = _compute_radius(best_c)
        
        if new_r > best_r:
            best_r = new_r
            no_improve_count = 0
            # Slightly increase step size if improving
            step_size *= 1.02
        else:
            no_improve_count += 1
            step_size *= 0.9
            if step_size < 1e-6:
                # Try a random kick to escape local minimum
                if np.random.rand() < 0.05:
                    kick_idx = np.random.choice(N, 3, replace=False)
                    best_c[kick_idx] += np.random.normal(0, 0.005, size=(3, 2))
                    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
                    best_r = _compute_radius(best_c)
                    step_size = 0.002
                else:
                    break
            if no_improve_count > 100:
                break
                
    return best_c, best_r


def _softmin_polish(c, time_budget=30.0):
    """
    Smooth surrogate optimization using soft-min.
    """
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    start = time.time()
    
    # Soft-min parameter
    beta = 50.0
    step_size = 0.001
    
    no_improve_count = 0
    
    while time.time() - start < time_budget:
        # Compute soft-min of pairwise distances
        d = cdist(best_c, best_c)
        np.fill_diagonal(d, np.inf)
        
        # Soft-min approximation: -log(sum(exp(-beta * d))) / beta
        # But this is expensive. Use a simpler approximation.
        # Find the K smallest distances and average them
        K = 5
        flat_d = d.flatten()
        flat_d = flat_d[flat_d > 0]
        if len(flat_d) < K:
            K = len(flat_d)
        
        # Get indices of K smallest
        idx = np.argpartition(flat_d, K)[:K]
        small_d = flat_d[idx]
        
        # Soft-min: weighted average with exponential weights
        weights = np.exp(-beta * (small_d - small_d.min()))
        weights /= weights.sum()
        soft_min_d = np.sum(weights * small_d)
        
        # Boundary soft-min
        boundary_dist = 0.5 - np.abs(best_c).max(axis=1)
        K_b = 3
        if len(boundary_dist) >= K_b:
            idx_b = np.argpartition(boundary_dist, K_b)[:K_b]
            small_b = boundary_dist[idx_b]
            weights_b = np.exp(-beta * (small_b - small_b.min()))
            weights_b /= weights_b.sum()
            soft_min_b = np.sum(weights_b * small_b)
        else:
            soft_min_b = boundary_dist.min()
        
        current_soft_r = min(soft_min_b, soft_min_d / 2.0)
        
        # Compute gradient of soft-min
        # For pairwise: gradient of soft_min_d wrt c_i is sum_j weights_ij * (c_i - c_j) / d_ij
        # This is complex. Use a simpler heuristic: move points that are in the K smallest pairs.
        
        forces = np.zeros_like(best_c)
        
        # Get the actual indices of the K smallest pairs
        d_flat_idx = np.argsort(flat_d)[:K]
        for idx_flat in d_flat_idx:
            i = idx_flat // N
            j = idx_flat % N
            if i == j:
                continue
            diff = best_c[i] - best_c[j]
            dist = d[i, j]
            if dist < 1e-10:
                continue
            force = diff / dist
            forces[i] += force
            forces[j] -= force
        
        # Boundary forces
        for idx_b in np.argwhere(boundary_dist < best_r + 1e-6):
            k = idx_b[0]
            if best_c[k, 0] > 0:
                forces[k, 0] -= 1.0
            else:
                forces[k, 0] += 1.0
            if best_c[k, 1] > 0:
                forces[k, 1] -= 1.0
            else:
                forces[k, 1] += 1.0
        
        # Normalize forces
        norms = np.linalg.norm(forces, axis=1)
        norms[norms == 0] = 1.0
        forces = forces / norms[:, np.newaxis]
        
        # Apply move
        new_c = best_c + step_size * forces
        new_c = np.clip(new_c, -0.5 + 1e-12, 0.5 - 1e-12)
        
        new_r = _compute_radius(new_c)
        
        if new_r > best_r:
            best_r = new_r
            best_c = new_c
            no_improve_count = 0
            step_size *= 1.03
        else:
            no_improve_count += 1
            step_size *= 0.9
            if step_size < 1e-7:
                if np.random.rand() < 0.05:
                    kick_idx = np.random.choice(N, 3, replace=False)
                    best_c[kick_idx] += np.random.normal(0, 0.005, size=(3, 2))
                    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
                    best_r = _compute_radius(best_c)
                    step_size = 0.001
                else:
                    break
            if no_improve_count > 150:
                break
                
    return best_c, best_r


def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing minimum radius."""
    global START_TIME, TIME_LIMIT
    
    np.random.seed(42)
    
    candidates = []
    
    # 1. Square grid
    c_square = _initial_square_grid(N)
    candidates.append(c_square)
    
    # 2. Hex grids with different offsets
    for ox in [0.0, 0.01, -0.01, 0.02, -0.02]:
        for oy in [0.0, 0.01, -0.01]:
            c_hex = _initial_hex_grid(N, offset_x=ox, offset_y=oy)
            candidates.append(c_hex)
    
    # 3. Perturbed hex
    c_hex_base = _initial_hex_grid(N)
    c_perturbed = c_hex_base + np.random.normal(0, 0.003, size=(N, 2))
    c_perturbed = np.clip(c_perturbed, -0.49, 0.49)
    candidates.append(c_perturbed)
    
    # Evaluate initial candidates
    best_c = None
    best_r = 0
    for c in candidates:
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c
    
    # Refine the best candidate using bottleneck polish
    elapsed = time.time() - START_TIME
    time_remaining = TIME_LIMIT - elapsed - 5  # 5s safety margin
    
    if time_remaining > 10:
        # Use bottleneck polish first (faster, more targeted)
        time_for_bottleneck = min(60.0, time_remaining * 0.6)
        c_refined, r_refined = _bottleneck_polish(best_c, time_budget=time_for_bottleneck)
        if r_refined > best_r:
            best_r = r_refined
            best_c = c_refined
        
        # Then use soft-min polish for further refinement
        elapsed = time.time() - START_TIME
        time_remaining = TIME_LIMIT - elapsed - 5
        if time_remaining > 10:
            time_for_softmin = min(40.0, time_remaining * 0.8)
            c_refined2, r_refined2 = _softmin_polish(best_c, time_budget=time_for_softmin)
            if r_refined2 > best_r:
                best_r = r_refined2
                best_c = c_refined2
    
    # Ensure valid output
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Final check
    if not np.all(np.isfinite(best_c)):
        return _initial_square_grid(N)
    
    return best_c
# EVOLVE-BLOCK-END
