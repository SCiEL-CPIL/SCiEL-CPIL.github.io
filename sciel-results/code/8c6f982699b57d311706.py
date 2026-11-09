# EVOLVE-BLOCK-START
import numpy as np
import time

def _vertex_score(distances_sorted):
    d = distances_sorted
    n = len(d)
    if n != 19:
        return 0.0
    if any(di < 0.05 for di in d):
        return 0.0
    diffs = d[1:] - d[:-1]
    min_diff = np.min(diffs)
    k = int(np.argmax(diffs == min_diff))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        four_d = d[0:4]
    elif k == 17:
        four_d = d[15:19]
    else:
        four_d = d[k-1:k+3]
    ratios = np.maximum(four_d / D, D / four_d)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
    return float(score)

def _compute_min_vertex_score(polygon):
    n = len(polygon)
    if n != 20:
        return 0.0
    if not np.all(np.isfinite(polygon)):
        return 0.0
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist
    
    # Check strict convexity
    for i in range(n):
        for j in range(i+1, n):
            if np.allclose(p[i], p[j], atol=1e-10):
                return 0.0
    
    cross_products = []
    for i in range(n):
        v1 = p[(i+1) % n] - p[i]
        v2 = p[(i+2) % n] - p[(i+1) % n]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        cross_products.append(cross)
    cross_products = np.array(cross_products)
    if np.any(np.abs(cross_products) < 1e-12):
        return 0.0
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return 0.0
    
    total_turn = 0.0
    for i in range(n):
        v1 = p[(i+1) % n] - p[i]
        v2 = p[(i+2) % n] - p[(i+1) % n]
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        turn = angle2 - angle1
        while turn > np.pi:
            turn -= 2 * np.pi
        while turn < -np.pi:
            turn += 2 * np.pi
        total_turn += turn
    if not np.isclose(abs(total_turn), 2 * np.pi, atol=1e-6):
        return 0.0
    
    min_score = 1.0
    for i in range(n):
        diffs = p - p[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        dists = np.delete(dists, i)
        dists_sorted = np.sort(dists)
        score = _vertex_score(dists_sorted)
        min_score = min(min_score, score)
    return min_score

def _regular_polygon():
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.stack([x, y], axis=1)

def _optimize_polygon(initial_poly, max_time=580.0):
    """
    Optimize polygon vertices to maximize minimum vertex score.
    Uses coordinate descent with adaptive step sizes.
    """
    start_time = time.time()
    current_poly = initial_poly.copy()
    current_score = _compute_min_vertex_score(current_poly)
    
    n = 20
    best_poly = current_poly.copy()
    best_score = current_score
    
    # Adaptive step size for coordinate descent
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < max_time:
        improved = False
        for i in range(n):
            if time.time() - start_time > max_time:
                break
            
            old_vertex = current_poly[i].copy()
            
            # Try moving in 8 directions
            directions = [
                np.array([1, 0]), np.array([-1, 0]),
                np.array([0, 1]), np.array([0, -1]),
                np.array([1, 1])/np.sqrt(2), np.array([-1, 1])/np.sqrt(2),
                np.array([1, -1])/np.sqrt(2), np.array([-1, -1])/np.sqrt(2)
            ]
            
            for direction in directions:
                new_poly = current_poly.copy()
                new_poly[i] = old_vertex + step_size * direction
                
                score = _compute_min_vertex_score(new_poly)
                if score > current_score + 1e-10:
                    current_poly = new_poly
                    current_score = score
                    improved = True
                    if score > best_score:
                        best_score = score
                        best_poly = current_poly.copy()
                    break
            
            # Also try opposite direction
            for direction in directions:
                new_poly = current_poly.copy()
                new_poly[i] = old_vertex - step_size * direction
                
                score = _compute_min_vertex_score(new_poly)
                if score > current_score + 1e-10:
                    current_poly = new_poly
                    current_score = score
                    improved = True
                    if score > best_score:
                        best_score = score
                        best_poly = current_poly.copy()
                    break
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Reduce step size when no improvement
            step_size *= 0.7
            if step_size < 1e-6 or no_improve_count > max_no_improve:
                break
    
    return best_poly, best_score

def _try_radial_modulation():
    """
    Try polygons with modulated radii to improve distance uniformity.
    The key insight: we want 4 distances from each vertex to be as equal as possible.
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)
    
    # Try various radial modulation patterns
    # Pattern: r_i = 1 + a * cos(m * theta_i)
    for a in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
        for m in [1, 2, 3, 4, 5]:
            radii = 1.0 + a * np.cos(m * base_angles)
            x = radii * np.cos(base_angles)
            y = radii * np.sin(base_angles)
            poly = np.stack([x, y], axis=1)
            score = _compute_min_vertex_score(poly)
            if score > best_score:
                best_score = score
                best_poly = poly.copy()
    
    # Try angular modulation
    for a in [0.01, 0.02, 0.03, 0.05]:
        for m in [1, 2, 3]:
            angles = base_angles + a * np.sin(m * base_angles)
            # Sort to maintain cyclic order
            order = np.argsort(angles)
            angles_sorted = angles[order]
            x = np.cos(angles_sorted)
            y = np.sin(angles_sorted)
            poly = np.stack([x, y], axis=1)
            score = _compute_min_vertex_score(poly)
            if score > best_score:
                best_score = score
                best_poly = poly.copy()
    
    return best_poly, best_score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 600s
    
    # Start with regular polygon
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)
    
    if time.time() > deadline:
        return best_poly
    
    # Try radial/angular modulation
    mod_poly, mod_score = _try_radial_modulation()
    if mod_score > best_score:
        best_score = mod_score
        best_poly = mod_poly
    
    if time.time() > deadline:
        return best_poly
    
    # Local optimization from best found so far
    opt_poly, opt_score = _optimize_polygon(best_poly, max_time=580.0)
    if opt_score > best_score:
        best_score = opt_score
        best_poly = opt_poly
    
    return best_poly

# EVOLVE-BLOCK-END
