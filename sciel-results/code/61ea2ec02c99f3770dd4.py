# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import sys

def _normalize(polygon):
    """
    Normalize the polygon: subtract centroid, divide by mean distance to centroid.
    Returns None if normalization is invalid (mean distance <= 0.01).
    """
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return None
    return centered / mean_dist

def _is_strictly_convex(polygon):
    """
    Check if the polygon is strictly convex and has no repeated points.
    Assumes vertices are in order.
    """
    n = len(polygon)
    if n < 3:
        return False
    
    # Vectorized cross product calculation
    p0 = polygon
    p1 = np.roll(polygon, -1, axis=0)
    p2 = np.roll(polygon, -2, axis=0)
    
    v1 = p1 - p0
    v2 = p2 - p1
    
    # Cross product in 2D: v1.x * v2.y - v1.y * v2.x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    # Nonzero turn
    if np.any(np.abs(cross) < 1e-10):
        return False
    
    # All turns must have the same sign
    signs = np.sign(cross)
    if not np.all(signs == signs[0]):
        return False
        
    return True

def _score_polygon_vectorized(polygon):
    """
    Calculate the score for a polygon.
    Returns the minimum vertex score.
    Returns -inf if invalid.
    """
    normalized = _normalize(polygon)
    if normalized is None:
        return -np.inf
        
    if not _is_strictly_convex(normalized):
        return -np.inf
        
    n = len(normalized)
    # Compute pairwise distances
    # dist[i, j] is distance between vertex i and j
    # We need distances from each vertex to all others, sorted.
    
    # Vectorized distance calculation
    # (n, 1, 2) - (1, n, 2) -> (n, n, 2)
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # dists[i] is the vector of distances from vertex i to all vertices j.
    # dists[i, i] is 0.
    
    # We need to sort the 19 distances from each vertex to the OTHER vertices.
    # So we set diagonal to inf, sort, and take first 19.
    np.fill_diagonal(dists, np.inf)
    sorted_dists = np.sort(dists, axis=1)[:, :19]
    
    # sorted_dists[i] = [d_0, d_1, ..., d_18] for vertex i
    
    # Check d_0 < 0.05
    min_dists = sorted_dists[:, 0]
    invalid_mask = min_dists < 0.05
    
    # Calculate gaps: d_{k+1} - d_k
    # We need to find k that minimizes the gap.
    # Gaps are defined for k=0..17 (between d_k and d_{k+1})
    gaps = np.diff(sorted_dists, axis=1) # shape (n, 18)
    
    # Find index of minimum gap
    k_indices = np.argmin(gaps, axis=1)
    
    # Determine the 4 distances to use for each vertex
    # If k == 0: d_0, d_1, d_2, d_3
    # If k == 17: d_15, d_16, d_17, d_18
    # Else: d_{k-1}, d_k, d_{k+1}, d_{k+2}
    
    # We will build an array of indices to select from sorted_dists
    # Shape (n, 4)
    
    # Initialize with zeros
    indices = np.zeros((n, 4), dtype=int)
    
    # Case 1: k == 0
    mask_0 = (k_indices == 0)
    indices[mask_0, 0] = 0
    indices[mask_0, 1] = 1
    indices[mask_0, 2] = 2
    indices[mask_0, 3] = 3
    
    # Case 2: k == 17
    mask_17 = (k_indices == 17)
    indices[mask_17, 0] = 15
    indices[mask_17, 1] = 16
    indices[mask_17, 2] = 17
    indices[mask_17, 3] = 18
    
    # Case 3: Else
    mask_else = ~(mask_0 | mask_17)
    k_else = k_indices[mask_else]
    # d_{k-1}
    indices[mask_else, 0] = k_else - 1
    # d_k
    indices[mask_else, 1] = k_else
    # d_{k+1}
    indices[mask_else, 2] = k_else + 1
    # d_{k+2}
    indices[mask_else, 3] = k_else + 2
    
    # Use advanced indexing to get the 4 distances for each vertex
    # sorted_dists[i, indices[i, j]]
    # We can do: sorted_dists[np.arange(n)[:, None], indices]
    row_indices = np.arange(n)[:, None]
    selected_dists = sorted_dists[row_indices, indices] # shape (n, 4)
    
    # Calculate D = (d_k + d_{k+1}) / 2
    # We need d_k and d_{k+1} specifically.
    # d_k is at index k_indices
    # d_{k+1} is at index k_indices + 1
    d_k = sorted_dists[np.arange(n), k_indices]
    d_k_plus_1 = sorted_dists[np.arange(n), k_indices + 1]
    D = (d_k + d_k_plus_1) / 2.0
    
    # Check D < 0.1
    invalid_mask |= (D < 0.1)
    
    # Calculate raw score
    # raw = mean over the 4 d's of max(d/D, D/d)
    # Let's compute ratio = d / D
    # ratio = max(ratio, 1/ratio)
    # Note: D might be 0? No, D >= 0.1.
    # Also d's are >= 0.05.
    
    ratios = selected_dists / D[:, None]
    # max(d/D, D/d) is equivalent to max(ratio, 1/ratio)
    # Since d, D > 0, this is symmetric.
    # If ratio < 1, 1/ratio > 1. If ratio > 1, ratio > 1.
    # So max(ratio, 1/ratio) = ratio if ratio >= 1 else 1/ratio.
    # This is equivalent to: np.where(ratio >= 1, ratio, 1/ratio)
    
    # However, we must be careful if D is 0, but we checked D >= 0.1.
    # Also selected_dists are distances, so > 0.
    
    raw_components = np.where(ratios >= 1.0, ratios, 1.0 / ratios)
    raw = np.mean(raw_components, axis=1)
    
    # vertex score = min(1, 1/raw)
    scores = np.minimum(1.0, 1.0 / raw)
    
    # Mark invalid vertices as -inf
    scores[invalid_mask] = -np.inf
    
    # Total score is minimum vertex score
    return np.min(scores)

def _angle_to_polygon(angles, radii):
    """
    Convert angles and radii to polygon vertices.
    angles: (n,) array of angles in [0, 2*pi)
    radii: (n,) array of positive radii
    """
    angles = np.sort(angles) % (2 * np.pi)
    # Ensure angles are in [0, 2*pi) and strictly increasing
    # If angles wrap around, we need to handle that
    # For convexity, we just need the order to be consistent
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _objective(params, n, radii):
    """
    Objective function for optimizer.
    params: flat array of n angles.
    radii: fixed radii array.
    Returns negative of the score (to minimize).
    """
    angles = params
    polygon = _angle_to_polygon(angles, radii)
    score = _score_polygon_vectorized(polygon)
    if np.isneginf(score) or np.isnan(score):
        return 1e10 # Large penalty
    return -score

def _objective_radii(params, n, angles):
    """
    Objective function for radii optimization.
    params: flat array of n radii (positive).
    angles: fixed angles array.
    Returns negative of the score (to minimize).
    """
    radii = np.abs(params) + 0.01  # Ensure positive
    polygon = _angle_to_polygon(angles, radii)
    score = _score_polygon_vectorized(polygon)
    if np.isneginf(score) or np.isnan(score):
        return 1e10 # Large penalty
    return -score

def _optimize_angles(angles, radii, n, max_iter=3000, xatol=1e-7, fatol=1e-9):
    """
    Optimize angles for fixed radii.
    """
    try:
        res = minimize(_objective, angles, args=(n, radii), method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True})
        return res.x
    except Exception:
        return angles

def _optimize_radii(angles, radii, n, max_iter=3000, xatol=1e-7, fatol=1e-9):
    """
    Optimize radii for fixed angles.
    """
    try:
        res = minimize(_objective_radii, radii - 0.01, args=(n, angles), method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True})
        new_radii = np.abs(res.x) + 0.01
        return new_radii
    except Exception:
        return radii

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 590.0  # 600s limit, keep 10s margin
    
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = -np.inf
    
    # Initial guess: Regular polygon
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    base_radii = np.ones(n)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Regular polygon
    initializations.append((base_angles.copy(), base_radii.copy()))
    
    # 2. Alternating radii patterns
    for pattern in range(5):
        angles = base_angles + rng.normal(0, 0.05, size=n)
        radii = 1.0 + rng.normal(0, 0.1, size=n)
        # Create some structured patterns
        if pattern == 0:
            radii[::2] += 0.1  # Even indices larger
        elif pattern == 1:
            radii[::3] += 0.15  # Every third larger
        elif pattern == 2:
            radii = 1.0 + 0.1 * np.sin(np.arange(n) * np.pi / 5)
        elif pattern == 3:
            radii = 1.0 + 0.15 * np.cos(np.arange(n) * 2 * np.pi / 7)
        elif pattern == 4:
            radii = 1.0 + rng.uniform(-0.1, 0.1, size=n)
        initializations.append((angles, radii))
    
    # 3. Wave patterns
    for freq in [2, 3, 5, 7, 10]:
        angles = base_angles + rng.normal(0, 0.03, size=n)
        radii = 1.0 + 0.1 * np.sin(freq * np.arange(n) * 2 * np.pi / n)
        initializations.append((angles, radii))
    
    # 4. Random with larger perturbations
    for _ in range(5):
        angles = base_angles + rng.normal(0, 0.2, size=n)
        radii = 1.0 + rng.uniform(-0.2, 0.2, size=n)
        radii = np.maximum(radii, 0.3)
        initializations.append((angles, radii))
    
    # Evaluate each initialization and run alternating optimization
    for init_idx, (angles, radii) in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
        
        # Evaluate initial score
        polygon = _angle_to_polygon(angles, radii)
        score = _score_polygon_vectorized(polygon)
        
        if not np.isneginf(score) and score > best_score:
            best_score = score
            best_params = (angles.copy(), radii.copy())
        
        # Run alternating optimization: optimize angles, then radii, repeat
        current_angles = angles.copy()
        current_radii = radii.copy()
        
        max_rounds = 3
        for round_idx in range(max_rounds):
            if time.time() > deadline - 10.0:
                break
            
            # Optimize angles
            new_angles = _optimize_angles(current_angles, current_radii, n,
                                         max_iter=2000, xatol=1e-8, fatol=1e-10)
            
            # Check if angles improved
            new_polygon = _angle_to_polygon(new_angles, current_radii)
            new_score = _score_polygon_vectorized(new_polygon)
            
            if not np.isneginf(new_score):
                current_angles = new_angles
                if new_score > best_score:
                    best_score = new_score
                    best_params = (current_angles.copy(), current_radii.copy())
            
            # Optimize radii
            new_radii = _optimize_radii(current_angles, current_radii, n,
                                       max_iter=2000, xatol=1e-8, fatol=1e-10)
            
            # Check if radii improved
            new_polygon = _angle_to_polygon(current_angles, new_radii)
            new_score = _score_polygon_vectorized(new_polygon)
            
            if not np.isneginf(new_score):
                current_radii = new_radii
                if new_score > best_score:
                    best_score = new_score
                    best_params = (current_angles.copy(), current_radii.copy())
        
        # Final high-precision polish if this was the best
        if best_params is not None and (init_idx == len(initializations) - 1 or score >= best_score - 1e-6):
            if time.time() > deadline - 15.0:
                break
            
            # Do a final tight optimization
            polish_angles = _optimize_angles(best_params[0], best_params[1], n,
                                            max_iter=5000, xatol=1e-9, fatol=1e-11)
            polish_radii = _optimize_radii(polish_angles, best_params[1], n,
                                          max_iter=5000, xatol=1e-9, fatol=1e-11)
            
            final_polygon = _angle_to_polygon(polish_angles, polish_radii)
            final_score = _score_polygon_vectorized(final_polygon)
            
            if not np.isneginf(final_score) and final_score > best_score:
                best_score = final_score
                best_params = (polish_angles, polish_radii)
    
    # Convert best params to polygon
    if best_params is not None:
        best_angles, best_radii = best_params
        final_polygon = _angle_to_polygon(best_angles, best_radii)
    else:
        # Fallback to regular polygon
        final_polygon = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    # Final validation
    final_score = _score_polygon_vectorized(final_polygon)
    if np.isneginf(final_score):
        # Fall back to regular polygon
        final_polygon = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    return final_polygon

# EVOLVE-BLOCK-END
