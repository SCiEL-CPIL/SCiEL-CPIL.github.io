"""Circle packing in unit square - 26 circles maximizing sum of radii.
Built on a Fast Marching / distance-field greedy placement approach.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.ndimage import distance_transform_edt, maximum_filter

N = 26

# EVOLVE-BLOCK-START

def _compute_distance_field(centers, radii, grid_n=200):
    """
    Compute the Euclidean distance field from the square boundary and
    subtract the influence of existing circles.
    Returns a 2D array of the 'free space' distance.
    """
    xs = np.linspace(0, 1, grid_n)
    ys = np.linspace(0, 1, grid_n)
    X, Y = np.meshgrid(xs, ys)
    
    # Distance to square boundary (positive inside)
    free_dist = np.minimum(np.minimum(X, 1 - X), np.minimum(Y, 1 - Y))
    
    # For each existing circle, compute distance to its surface and take minimum
    n_circ = len(centers)
    if n_circ > 0:
        for i in range(n_circ):
            if radii[i] <= 0:
                continue
            cx, cy = centers[i]
            d_to_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            dist_to_surface = d_to_center - radii[i]
            free_dist = np.minimum(free_dist, dist_to_surface)
    
    free_dist = np.maximum(free_dist, 0.0)
    return free_dist, xs, ys

def _max_radius_at(x, y, centers, radii):
    """Compute the maximum radius for a circle at position (x,y)."""
    r = min(x, 1 - x, y, 1 - y)
    n = len(centers)
    if n > 0:
        dx = x - centers[:, 0]
        dy = y - centers[:, 1]
        dists = np.sqrt(dx * dx + dy * dy)
        r = min(r, np.min(dists - radii))
    return max(r, 0.0)

def _max_radius_at_vec(x, y, pos, radii):
    """Vectorized max radius computation for a single position."""
    r = min(x, 1.0 - x, y, 1.0 - y)
    n = len(pos)
    if n > 0:
        dx = x - pos[:, 0]
        dy = y - pos[:, 1]
        dists = np.sqrt(dx * dx + dy * dy)
        r = min(r, np.min(dists - radii))
    return max(r, 0.0)

def _refine_position(x, y, centers, radii, max_iter=40):
    """
    Refine a position to find the exact maximum inscribed circle radius
    at that location using iterative gradient ascent with adaptive steps.
    """
    for _ in range(max_iter):
        r = _max_radius_at(x, y, centers, radii)
        
        if r <= 0:
            return x, y, 0.001
        
        best_r = r
        best_x, best_y = x, y
        # Adaptive step sizes: start large, shrink
        step_sizes = [0.06, 0.03, 0.015, 0.008, 0.004, 0.002, 0.001, 0.0005]
        
        for step in step_sizes:
            for angle in np.linspace(0, 2*np.pi, 32, endpoint=False):
                nx = x + step * np.cos(angle)
                ny = y + step * np.sin(angle)
                
                if nx < 0.001 or nx > 0.999 or ny < 0.001 or ny > 0.999:
                    continue
                
                nr = _max_radius_at(nx, ny, centers, radii)
                
                if nr > best_r + 1e-8:
                    best_r = nr
                    best_x, best_y = nx, ny
        
        if abs(best_x - x) < 1e-7 and abs(best_y - y) < 1e-7:
            break
        x, y = best_x, best_y
    
    r = _max_radius_at(x, y, centers, radii)
    return x, y, max(r, 0.001)

def _find_local_maxima(free_dist, min_distance=5):
    """
    Find local maxima in the distance field using a sliding window maximum filter.
    Returns list of (x, y, r) candidates where r is the local max value.
    """
    grid_n = free_dist.shape[0]
    
    size = min_distance * 2 + 1
    max_filter = maximum_filter(free_dist, size=size)
    local_max_mask = (free_dist == max_filter) & (free_dist > 0.005)
    
    rows, cols = np.where(local_max_mask)
    candidates = []
    for r, c in zip(rows, cols):
        x = c / (grid_n - 1)
        y = r / (grid_n - 1)
        radius = free_dist[r, c]
        candidates.append((x, y, radius))
    
    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates

def _find_multiple_candidates(free_dist, centers, radii, n_candidates=20, min_distance=3):
    """
    Find multiple good candidate positions for placing the next circle.
    Uses the distance field but also considers actual geometric constraints.
    """
    grid_n = free_dist.shape[0]
    
    size = min_distance * 2 + 1
    max_filter = maximum_filter(free_dist, size=size)
    local_max_mask = (free_dist == max_filter) & (free_dist > 0.003)
    
    rows, cols = np.where(local_max_mask)
    candidates = []
    
    for r, c in zip(rows, cols):
        x = c / (grid_n - 1)
        y = r / (grid_n - 1)
        est_radius = free_dist[r, c]
        candidates.append((x, y, est_radius))
    
    # Sort by estimated radius
    candidates.sort(key=lambda t: t[2], reverse=True)
    
    # Refine top candidates and keep the best ones
    refined = []
    for x, y, est_r in candidates[:100]:
        rx, ry, actual_r = _refine_position(x, y, centers, radii)
        if actual_r > 0.004:
            refined.append((rx, ry, actual_r))
    
    # Remove duplicates (positions too close together)
    unique_refined = []
    for x, y, r in refined:
        is_unique = True
        for ux, uy, ur in unique_refined:
            if np.sqrt((x - ux)**2 + (y - uy)**2) < 0.012:
                is_unique = False
                break
        if is_unique:
            unique_refined.append((x, y, r))
    
    unique_refined.sort(key=lambda t: t[2], reverse=True)
    return unique_refined[:n_candidates]

def _greedy_distance_field_placement(grid_n=200, n_circles=N, seed=42):
    """
    Greedy placement using distance field:
    1. Compute initial distance field from boundary
    2. Find local maxima, place largest circle there
    3. Update distance field by subtracting circle influence
    4. Repeat until all circles placed or no more valid locations
    """
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    # Initial distance field
    free_dist, xs, ys = _compute_distance_field(np.empty(0), np.empty(0), grid_n)
    
    placed_centers = []
    placed_radii = []
    
    for i in range(n_circles):
        # Find multiple candidates and pick the best
        min_dist = max(2, int(0.025 * grid_n))
        candidates = _find_multiple_candidates(free_dist, placed_centers, placed_radii, 
                                                n_candidates=15, min_distance=min_dist)
        
        if not candidates:
            # Try with smaller min_distance
            candidates = _find_local_maxima(free_dist, min_distance=1)
            if candidates:
                candidates = [candidates[0]]
        
        if not candidates:
            # No more valid locations; use small fallback
            centers[i] = [0.5, 0.5]
            radii[i] = 0.005
            placed_centers.append(centers[i])
            placed_radii.append(radii[i])
            continue
        
        # Pick the best candidate (largest radius)
        best_x, best_y, best_r = candidates[0]
        
        # Verify and adjust: ensure the circle fits
        actual_r = best_r * 0.998
        
        # Check against boundary
        max_r_boundary = min(best_x, 1 - best_x, best_y, 1 - best_y)
        actual_r = min(actual_r, max_r_boundary * 0.999)
        
        # Check against existing circles
        for j in range(len(placed_centers)):
            dx = best_x - placed_centers[j][0]
            dy = best_y - placed_centers[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r_circle = dist - placed_radii[j]
            if max_r_circle < actual_r:
                actual_r = max_r_circle * 0.999
        
        actual_r = max(actual_r, 0.005)
        
        centers[i] = [best_x, best_y]
        radii[i] = actual_r
        placed_centers.append(centers[i])
        placed_radii.append(radii[i])
        
        # Update distance field
        free_dist, xs, ys = _compute_distance_field(
            np.array(placed_centers), np.array(placed_radii), grid_n
        )
    
    return centers, radii

def _greedy_with_refinement(grid_n=250, n_circles=N, seed=42):
    """
    Enhanced greedy: after initial placement, try to refine positions
    by local search to increase radii.
    """
    centers, radii = _greedy_distance_field_placement(grid_n, n_circles, seed)
    
    # Local refinement: for each circle, try to grow it by moving it slightly
    for pass_num in range(5):
        for i in range(n_circles):
            improved = True
            while improved:
                improved = False
                best_r = radii[i]
                best_x, best_y = centers[i]
                
                # Multi-scale local search
                for step in [0.05, 0.025, 0.012, 0.006, 0.003, 0.0015, 0.0008]:
                    for angle in np.linspace(0, 2*np.pi, 24, endpoint=False):
                        nx = centers[i][0] + step * np.cos(angle)
                        ny = centers[i][1] + step * np.sin(angle)
                        
                        # Check boundary
                        if nx < 0.001 or nx > 0.999 or ny < 0.001 or ny > 0.999:
                            continue
                        
                        max_r = min(nx, 1 - nx, ny, 1 - ny)
                        # Check other circles
                        for j in range(n_circles):
                            if j == i:
                                continue
                            d = np.sqrt((nx - centers[j][0])**2 + (ny - centers[j][1])**2)
                            max_r = min(max_r, d - radii[j])
                        
                        if max_r > best_r + 1e-7:
                            best_r = max_r
                            best_x, best_y = nx, ny
                            improved = True
                
                if improved:
                    centers[i] = [best_x, best_y]
                    radii[i] = best_r * 0.9995
    
    # Second pass: try to grow all circles simultaneously
    for _ in range(30):
        any_grew = False
        for i in range(n_circles):
            # Try to grow circle i by shrinking others slightly if needed
            target_r = radii[i] * 1.003
            # Check if we can grow without violating constraints
            can_grow = True
            for j in range(n_circles):
                if j == i:
                    continue
                d = np.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
                if d < target_r + radii[j]:
                    can_grow = False
                    break
            if can_grow and min(centers[i][0], 1-centers[i][0], centers[i][1], 1-centers[i][1]) >= target_r:
                radii[i] = target_r
                any_grew = True
        if not any_grew:
            break
    
    return centers, radii

def _initial_hexagonal():
    """Hexagonal lattice initialization as fallback."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        
        if k >= N:
            break
            
    while k < N:
        rng = np.random.RandomState(k * 7 + 13)
        for _ in range(200):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0])**2 + (y - pos[j, 1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.015:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.98
                k += 1
                break
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.008
            k += 1
            
    return pos, radii

def _initial_corner_focus():
    """Initialize circles focusing on corners and edges where larger circles can fit."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with corner circles
    corner_positions = [
        (0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)
    ]
    corner_radii = [0.14, 0.14, 0.14, 0.14]
    
    for i in range(4):
        pos[i] = corner_positions[i]
        radii[i] = corner_radii[i]
    
    # Edge centers
    edge_positions = [
        (0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5)
    ]
    edge_radii = [0.07, 0.07, 0.07, 0.07]
    
    for i in range(4, 8):
        pos[i] = edge_positions[i-4]
        radii[i] = edge_radii[i-4]
    
    # Center region
    remaining = N - 8
    rng = np.random.RandomState(42)
    for k in range(8, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        for _ in range(500):
            x = rng.uniform(0.2, 0.8)
            y = rng.uniform(0.2, 0.8)
            r = min(x, 1-x, y, 1-y)
            for j in range(k):
                d = np.sqrt((x - pos[j,0])**2 + (y - pos[j,1])**2)
                r = min(r, d - radii[j])
            if r > best_r:
                best_x, best_y, best_r = x, y, r
        if best_r > 0.01:
            pos[k] = [best_x, best_y]
            radii[k] = best_r * 0.98
        else:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
    
    return pos, radii

def _initial_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(200):
        valid = True
        
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _objective(x, penalty_weight=100.0):
    """
    Objective: minimize -sum(radii) + penalty * violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _simulated_annealing(pos, radii, deadline, n_iter=8000):
    """
    Simulated annealing to improve the packing.
    Tries random moves and accepts worse solutions with decreasing probability.
    """
    pos = pos.copy()
    radii = radii.copy()
    cur_score = np.sum(radii)
    best_score = cur_score
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    T = 0.04  # Initial temperature
    T_min = 0.0003
    cooling_rate = 0.9996
    
    for it in range(n_iter):
        if time.time() > deadline - 2:
            break
        
        # Pick a random circle to move
        i = np.random.randint(N)
        old_x, old_y = pos[i]
        old_r = radii[i]
        
        # Random perturbation - position move with adaptive scale
        scale = T * 0.7
        dx = np.random.normal(0, scale)
        dy = np.random.normal(0, scale)
        
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Determine max radius at new position
        new_r = _max_radius_at(new_x, new_y, pos, radii)
        # Try to keep similar radius or grow slightly
        new_r = min(new_r, old_r * 1.15)
        new_r = max(new_r, 0.004)
        
        # Check if move is valid
        valid = True
        # Boundary check
        if new_x - new_r < -1e-7 or new_x + new_r > 1 + 1e-7:
            valid = False
        if new_y - new_r < -1e-7 or new_y + new_r > 1 + 1e-7:
            valid = False
        
        if valid:
            # Overlap check
            for j in range(N):
                if j == i:
                    continue
                d = np.sqrt((new_x - pos[j,0])**2 + (new_y - pos[j,1])**2)
                if d < new_r + radii[j] - 1e-7:
                    valid = False
                    break
        
        if valid:
            new_score = cur_score - old_r + new_r
            delta = new_score - cur_score
            
            if delta > 0:
                # Always accept better solutions
                pos[i] = [new_x, new_y]
                radii[i] = new_r
                cur_score = new_score
                if cur_score > best_score:
                    best_pos = pos.copy()
                    best_radii = radii.copy()
                    best_score = cur_score
            else:
                # Accept worse solutions with probability exp(delta/T)
                if T > T_min:
                    ratio = np.exp(delta / T)
                    if np.random.random() < ratio:
                        pos[i] = [new_x, new_y]
                        radii[i] = new_r
                        cur_score = new_score
        
        T = max(T * cooling_rate, T_min)
    
    return best_pos, best_radii

def _pair_interact_optimize(pos, radii, deadline, max_rounds=50):
    """
    Optimize by considering pairs of circles: for each pair, try to move
    one or both to increase the minimum radius in the pair.
    """
    pos = pos.copy()
    radii = radii.copy()
    score = np.sum(radii)
    
    for round_num in range(max_rounds):
        if time.time() > deadline - 3:
            break
        
        improved = False
        for i in range(N):
            for j in range(i + 1, N):
                # Try moving circle i to give more room to circle j
                for direction in ['i', 'j', 'both']:
                    if direction == 'i':
                        # Move i away from j
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < 1e-8:
                            continue
                        # Try small moves in the direction away from j
                        for step in [0.005, 0.01, 0.02, 0.03]:
                            nx = pos[i, 0] + step * dx / dist
                            ny = pos[i, 1] + step * dy / dist
                            if nx < 0.001 or nx > 0.999 or ny < 0.001 or ny > 0.999:
                                continue
                            new_r_i = _max_radius_at(nx, ny, pos, radii)
                            # Check if this helps j
                            new_dist = np.sqrt((pos[j, 0] - nx)**2 + (pos[j, 1] - ny)**2)
                            new_r_j_max = min(new_dist - radii[i], 
                                              min(pos[j, 0], 1-pos[j, 0], pos[j, 1], 1-pos[j, 1]))
                            new_r_j = min(radii[j] * 1.01, new_r_j_max)
                            new_r_j = max(new_r_j, 0.004)
                            
                            # Check validity
                            valid = True
                            for k in range(N):
                                if k == i:
                                    continue
                                d = np.sqrt((nx - pos[k, 0])**2 + (ny - pos[k, 1])**2)
                                if d < new_r_i + radii[k] - 1e-7:
                                    valid = False
                                    break
                            if valid:
                                new_score = score - radii[i] - radii[j] + new_r_i + new_r_j
                                if new_score > score + 1e-8:
                                    pos[i] = [nx, ny]
                                    radii[i] = new_r_i
                                    radii[j] = new_r_j
                                    score = new_score
                                    improved = True
                                    break
                    elif direction == 'j':
                        # Move j away from i
                        dx = pos[j, 0] - pos[i, 0]
                        dy = pos[j, 1] - pos[i, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < 1e-8:
                            continue
                        for step in [0.005, 0.01, 0.02, 0.03]:
                            nx = pos[j, 0] + step * dx / dist
                            ny = pos[j, 1] + step * dy / dist
                            if nx < 0.001 or nx > 0.999 or ny < 0.001 or ny > 0.999:
                                continue
                            new_r_j = _max_radius_at(nx, ny, pos, radii)
                            new_dist = np.sqrt((pos[i, 0] - nx)**2 + (pos[i, 1] - ny)**2)
                            new_r_i_max = min(new_dist - radii[j],
                                              min(pos[i, 0], 1-pos[i, 0], pos[i, 1], 1-pos[i, 1]))
                            new_r_i = min(radii[i] * 1.01, new_r_i_max)
                            new_r_i = max(new_r_i, 0.004)
                            
                            valid = True
                            for k in range(N):
                                if k == j:
                                    continue
                                d = np.sqrt((nx - pos[k, 0])**2 + (ny - pos[k, 1])**2)
                                if d < new_r_j + radii[k] - 1e-7:
                                    valid = False
                                    break
                            if valid:
                                new_score = score - radii[i] - radii[j] + new_r_i + new_r_j
                                if new_score > score + 1e-8:
                                    pos[j] = [nx, ny]
                                    radii[j] = new_r_j
                                    radii[i] = new_r_i
                                    score = new_score
                                    improved = True
                                    break
        if not improved:
            break
    
    return pos, radii, score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Strategy 1: Distance field greedy placement
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    # Try different grid resolutions
    for grid_n in [180, 220, 260, 300]:
        if time.time() > deadline - 55:
            break
        try:
            pos_gd, rad_gd = _greedy_distance_field_placement(grid_n, N)
            pos_gd, rad_gd = _post_process(pos_gd, rad_gd)
            if _is_valid(pos_gd, rad_gd):
                score = np.sum(rad_gd)
                if score > best_score:
                    best_score = score
                    best_pos = pos_gd.copy()
                    best_radii = rad_gd.copy()
        except Exception:
            pass
    
    # Strategy 2: Greedy with refinement
    if time.time() < deadline - 35:
        try:
            pos_gr, rad_gr = _greedy_with_refinement(220, N)
            pos_gr, rad_gr = _post_process(pos_gr, rad_gr)
            if _is_valid(pos_gr, rad_gr):
                score = np.sum(rad_gr)
                if score > best_score:
                    best_score = score
                    best_pos = pos_gr.copy()
                    best_radii = rad_gr.copy()
        except Exception:
            pass
    
    # Strategy 3: Corner-focused initialization
    if time.time() < deadline - 30:
        try:
            pos_c, rad_c = _initial_corner_focus()
            pos_c, rad_c = _post_process(pos_c, rad_c)
            if _is_valid(pos_c, rad_c):
                score = np.sum(rad_c)
                if score > best_score:
                    best_score = score
                    best_pos = pos_c.copy()
                    best_radii = rad_c.copy()
        except Exception:
            pass
    
    # Strategy 4: Random initializations
    for seed in [42, 123, 789, 555]:
        if time.time() > deadline - 25:
            break
        try:
            pos_r, rad_r = _initial_random(seed)
            if _is_valid(pos_r, rad_r):
                score = np.sum(rad_r)
                if score > best_score:
                    best_score = score
                    best_pos = pos_r.copy()
                    best_radii = rad_r.copy()
        except Exception:
            pass
    
    # If no valid solution found, use a safe default
    if best_pos is None:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # Polish with L-BFGS-B
    if time.time() < deadline - 12:
        try:
            pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
            pos_pol, rad_pol = _post_process(pos_pol, rad_pol)
            if _is_valid(pos_pol, rad_pol):
                score = np.sum(rad_pol)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pol.copy()
                    best_radii = rad_pol.copy()
        except Exception:
            pass
    
    # Pair interaction optimization
    if time.time() < deadline - 10:
        try:
            pos_pi, rad_pi, score_pi = _pair_interact_optimize(best_pos, best_radii, deadline, max_rounds=40)
            pos_pi, rad_pi = _post_process(pos_pi, rad_pi)
            if _is_valid(pos_pi, rad_pi):
                score = np.sum(rad_pi)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pi.copy()
                    best_radii = rad_pi.copy()
        except Exception:
            pass
    
    # Simulated annealing for further improvement
    if time.time() < deadline - 8:
        try:
            sa_pos, sa_radii = _simulated_annealing(best_pos, best_radii, deadline, n_iter=5000)
            sa_pos, sa_radii = _post_process(sa_pos, sa_radii)
            if _is_valid(sa_pos, sa_radii):
                score = np.sum(sa_radii)
                if score > best_score:
                    best_score = score
                    best_pos = sa_pos.copy()
                    best_radii = sa_radii.copy()
        except Exception:
            pass
    
    # Final polish pass
    if time.time() < deadline - 5:
        try:
            pos_pol2, rad_pol2 = _polish(best_pos, best_radii, deadline)
            pos_pol2, rad_pol2 = _post_process(pos_pol2, rad_pol2)
            if _is_valid(pos_pol2, rad_pol2):
                score = np.sum(rad_pol2)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pol2.copy()
                    best_radii = rad_pol2.copy()
        except Exception:
            pass
    
    # Another pair interaction pass
    if time.time() < deadline - 3:
        try:
            pos_pi2, rad_pi2, score_pi2 = _pair_interact_optimize(best_pos, best_radii, deadline, max_rounds=30)
            pos_pi2, rad_pi2 = _post_process(pos_pi2, rad_pi2)
            if _is_valid(pos_pi2, rad_pi2):
                score = np.sum(rad_pi2)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pi2.copy()
                    best_radii = rad_pi2.copy()
        except Exception:
            pass
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
