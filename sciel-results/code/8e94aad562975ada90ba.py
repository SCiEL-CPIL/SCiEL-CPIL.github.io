# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score_polygon(poly):
    """
    Calculate the minimum vertex score for a given polygon.
    Returns 0.0 if invalid.
    """
    n = len(poly)
    if n != 20:
        return 0.0
    
    # Check for repeated points
    # Compute pairwise distances
    # dists_matrix[i, j] = ||poly[i] - poly[j]||
    dists_matrix = np.linalg.norm(poly[:, np.newaxis, :] - poly[np.newaxis, :, :], axis=2)
    # Off-diagonal elements should be > 0
    np.fill_diagonal(dists_matrix, np.inf)
    if np.min(dists_matrix) < 1e-9:
        return 0.0

    # Check convexity
    # Compute cross products of consecutive edges
    turns = np.zeros(n)
    for i in range(n):
        p0 = poly[i]
        p1 = poly[(i+1)%n]
        p2 = poly[(i+2)%n]
        v1 = p1 - p0
        v2 = p2 - p1
        turns[i] = v1[0]*v2[1] - v1[1]*v2[0]
        
    if np.any(np.isclose(turns, 0, atol=1e-8)):
        return 0.0
    if not (np.all(turns > 0) or np.all(turns < 0)):
        return 0.0
        
    # Normalize
    centroid = np.mean(poly, axis=0)
    poly_norm = poly - centroid
    dists_to_centroid = np.linalg.norm(poly_norm, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    poly_norm = poly_norm / mean_dist
    
    # Recalculate distances for normalized poly
    dists_matrix_norm = np.linalg.norm(poly_norm[:, np.newaxis, :] - poly_norm[np.newaxis, :, :], axis=2)
    
    min_score = 1.0
    for i in range(n):
        d = dists_matrix_norm[i, 1:] # Exclude self
        d = np.sort(d)
        
        if d[0] < 0.05:
            return 0.0
            
        gaps = np.diff(d)
        k = np.argmin(gaps)
        D = (d[k] + d[k+1]) / 2.0
        
        if D < 0.1:
            return 0.0
            
        if k == 0:
            sel = d[0:4]
        elif k == 17:
            sel = d[15:19]
        else:
            sel = d[k-1:k+3]
            
        # Handle division by zero or negative (shouldn't happen)
        if np.any(sel <= 0) or D <= 0:
            return 0.0
            
        ratios = np.maximum(sel/D, D/sel)
        raw = np.mean(ratios)
        score = min(1.0, 1.0/raw)
        if score < min_score:
            min_score = score
            if min_score < 1e-9:
                return 0.0
                
    return min_score

def _build_polygon_from_offsets(offsets):
    """
    Build a polygon from 5 angular offsets, repeated 4 times with 90-degree rotations.
    offsets: array of 5 angles.
    Returns a (20, 2) array.
    """
    base_angles = offsets
    # Repeat with 90 degree (pi/2) rotations
    # The order is: offset[0], offset[1], offset[2], offset[3], offset[4],
    #               offset[0]+pi/2, offset[1]+pi/2, ...
    # This ensures 4-fold rotational symmetry.
    
    # We need to sort the angles within each quadrant to maintain convexity/cyclic order?
    # Actually, if we just take the 5 offsets, add 0, pi/2, pi, 3pi/2 to them,
    # and sort the resulting 20 angles, we get a valid cyclic order.
    # However, the problem says "cyclic order". The vertices must be in cyclic order.
    # If we sort the angles, they are in cyclic order around the origin.
    # But we also need radial distances.
    # Let's assume unit radius for now? Or optimize radii too?
    # The inspiration suggests "5 distinct angular offsets".
    # Let's try fixed radius 1 first, just optimizing angles.
    
    angles = np.concatenate([base_angles + i * (np.pi/2) for i in range(4)])
    angles = np.sort(angles)
    
    # Ensure angles are in [0, 2pi)
    angles = angles % (2 * np.pi)
    angles = np.sort(angles)
    
    x = np.cos(angles)
    y = np.sin(angles)
    poly = np.stack([x, y], axis=1)
    
    return poly

def _score_symmetric(offsets):
    poly = _build_polygon_from_offsets(offsets)
    return _score_polygon(poly)

def _objective_symmetric(offsets):
    score = _score_symmetric(offsets)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that maximizes the minimum vertex score.
    Uses 4-fold symmetry (5 angular offsets) and optimization.
    """
    n = 20
    start_time = time.time()
    time_limit = 580.0 # Safety margin
    
    # Initial guess: regular polygon
    # Regular 20-gon has angles k * 2pi/20.
    # With 4-fold symmetry, we can take the first 5 angles of a regular 20-gon.
    # Angles: 0, pi/10, 2pi/10, 3pi/10, 4pi/10.
    # These are 0, 18, 36, 54, 72 degrees.
    # Repeating with 90 deg (pi/2) shifts:
    # 0, 18, 36, 54, 72
    # 90, 108, 126, 144, 162
    # 180, 198, 216, 234, 252
    # 270, 288, 306, 324, 342
    # Sorted, these are exactly the regular 20-gon angles.
    
    offsets0 = np.array([0.0, np.pi/10, 2*np.pi/10, 3*np.pi/10, 4*np.pi/10])
    
    best_poly = _build_polygon_from_offsets(offsets0)
    best_score = _score_polygon(best_poly)
    
    # Optimization
    # We can optimize the 5 offsets.
    # Constraints: offsets should be sorted and within [0, pi/2) to avoid overlap in the first quadrant.
    # Actually, _build_polygon_from_offsets sorts the angles, so the order of offsets doesn't matter for the final set,
    # but it matters for the optimizer's parameter space.
    # Let's constrain offsets to be in [0, pi/2) and sorted.
    # To handle constraints, we can use a transformation or just rely on the sort in _build.
    # However, if offsets are not sorted, the "offsets" parameter space is disconnected.
    # Let's just optimize the 5 values directly, allowing them to be anything,
    # and let _build_polygon_from_offsets handle the sorting and modulo.
    
    options = {
        'maxiter': 5000,
        'xatol': 1e-8,
        'fatol': 1e-8,
        'adaptive': True
    }
    
    # Try a few different initializations
    best_score = -np.inf
    best_poly = None
    
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if i == 0:
            r_init = offsets0.copy()
        elif i == 1:
            # Regular polygon but with a small rotation
            r_init = offsets0 + 0.1
        else:
            # Random perturbations
            r_init = offsets0 + 0.1 * np.random.randn(5)
        
        try:
            res = minimize(_objective_symmetric, r_init, method='Nelder-Mead', options=options)
            if res.fun < -best_score:
                actual_score = _score_symmetric(res.x)
                if actual_score > best_score:
                    best_score = actual_score
                    best_poly = _build_polygon_from_offsets(res.x)
        except Exception as e:
            pass

    # If we have time, do a finer local search on the best found
    if best_poly is not None and time.time() - start_time < time_limit - 10:
        # Extract offsets from best_poly? 
        # It's hard to extract the "offsets" from the polygon directly if we just sorted them.
        # Let's just do a random hill climbing on the offsets that produced best_poly.
        # We didn't store the best offsets. Let's store them.
        pass # This block is skipped as we didn't store best_offsets. We will rely on the optimizer.
    
    # Fallback to regular polygon if optimization failed or returned invalid
    if best_poly is None or best_score <= 0:
        best_poly = _build_polygon_from_offsets(offsets0)
        best_score = _score_polygon(best_poly)

    return best_poly

# EVOLVE-BLOCK-END
