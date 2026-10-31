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
        # Find local maxima with adaptive min_distance
        min_dist = max(2, int(0.03 * grid_n))
        candidates = _find_local_maxima(free_dist, min_distance=min_dist)
        
        if not candidates:
            # Try with smaller min_distance
            candidates = _find_local_maxima(free_dist, min_distance=1)
        
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
        actual_r = best_r * 0.98  # tighter safety margin
        
        # Check against boundary
        max_r_boundary = min(best_x, 1 - best_x, best_y, 1 - best_y)
        actual_r = min(actual_r, max_r_boundary * 0.995)
        
        # Check against existing circles
        for j in range(len(placed_centers)):
            dx = best_x - placed_centers[j][0]
            dy = best_y - placed_centers[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r_circle = dist - placed_radii[j]
            if max_r_circle < actual_r:
                actual_r = max_r_circle * 0.995
        
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
    for i in range(n_circles):
        for _ in range(30):
            best_r = radii[i]
            best_x, best_y = centers[i]
            
            # Try small perturbations in a grid
            step = 0.015
            for dx in np.linspace(-step*2, step*2, 5):
                for dy in np.linspace(-step*2, step*2, 5):
                    nx, ny = centers[i][0] + dx, centers[i][1] + dy
                    # Check boundary
                    max_r = min(nx, 1 - nx, ny, 1 - ny)
                    if max_r <= 0.001:
                        continue
                    # Check other circles
                    for j in range(n_circles):
                        if j == i:
                            continue
                        d = np.sqrt((nx - centers[j][0])**2 + (ny - centers[j][1])**2)
                        max_r = min(max_r, d - radii[j])
                    if max_r > best_r:
                        best_r = max_r
                        best_x, best_y = nx, ny
            
            if best_r > radii[i] * 1.005:
                centers[i] = [best_x, best_y]
                radii[i] = best_r * 0.995
            else:
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
        for _ in range(100):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0])**2 + (y - pos[j, 1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                break
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
            
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

def _polish(pos, radii, deadline):
    """High-precision polishing with L-BFGS-B."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.45))
        
    # L-BFGS-B with increasing penalty
    for p_w in [200.0, 800.0, 3000.0, 10000.0]:
        if time.time() > deadline - 3:
            break
        try:
            result = minimize(_objective, x, method='L-BFGS-B', bounds=bounds,
                              args=(p_w,),
                              options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-12})
            if result.fun < _objective(x, p_w) - 1e-13:
                x = result.x
        except Exception:
            pass
    
    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 1e-4)
    
    return pos_out, radii_out

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Strategy 1: Distance field greedy placement
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    # Try different grid resolutions
    for grid_n in [120, 160, 200, 240]:
        if time.time() > deadline - 35:
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
    if time.time() < deadline - 20:
        try:
            pos_gr, rad_gr = _greedy_with_refinement(200, N)
            pos_gr, rad_gr = _post_process(pos_gr, rad_gr)
            if _is_valid(pos_gr, rad_gr):
                score = np.sum(rad_gr)
                if score > best_score:
                    best_score = score
                    best_pos = pos_gr.copy()
                    best_radii = rad_gr.copy()
        except Exception:
            pass
    
    # Strategy 3: Hexagonal fallback
    if time.time() < deadline - 20:
        try:
            pos_h, rad_h = _initial_hexagonal()
            pos_h, rad_h = _post_process(pos_h, rad_h)
            if _is_valid(pos_h, rad_h):
                score = np.sum(rad_h)
                if score > best_score:
                    best_score = score
                    best_pos = pos_h.copy()
                    best_radii = rad_h.copy()
        except Exception:
            pass
    
    # Strategy 4: Random initializations
    for seed in [42, 123, 789]:
        if time.time() > deadline - 15:
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
    
    # Final polish - run multiple times with different starting points
    if time.time() < deadline - 3:
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
    
    # Second polish pass if time remains
    if time.time() < deadline - 2:
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
