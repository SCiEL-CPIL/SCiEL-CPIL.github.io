# EVOLVE-BLOCK-START
import numpy as np
import time


def _normalize(polygon):
    """Subtract centroid and divide by mean distance to centroid."""
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = dists.mean()
    if mean_dist <= 0.01:
        return centered
    return centered / mean_dist


def _is_valid_convex(polygon):
    """Check strictly convex: no repeated points, all turns same sign nonzero, total turning +-2*pi."""
    n = len(polygon)
    if n < 3:
        return False
    # Check no repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-10:
                return False
    # Compute cross products of consecutive edges
    cross_prods = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        cross_prods.append(cross)
    cross_prods = np.array(cross_prods)
    # All must be nonzero and same sign
    if np.any(np.abs(cross_prods) < 1e-12):
        return False
    if np.any(cross_prods > 0) and np.any(cross_prods < 0):
        return False
    # Total turning should be +-2*pi
    # Compute turning angles
    total_turn = 0.0
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        angle1 = np.arctan2(e1[1], e1[0])
        angle2 = np.arctan2(e2[1], e2[0])
        d_angle = angle2 - angle1
        # Normalize to [-pi, pi]
        while d_angle > np.pi:
            d_angle -= 2 * np.pi
        while d_angle < -np.pi:
            d_angle += 2 * np.pi
        total_turn += d_angle
    if abs(abs(total_turn) - 2 * np.pi) > 1e-6:
        return False
    return True


def _vertex_score(vertex, polygon):
    """Compute the score for a single vertex."""
    n = len(polygon)
    # Distances to all other vertices
    diffs = polygon - vertex
    dists = np.linalg.norm(diffs, axis=1)
    # Set self distance to inf
    idx = np.argmin(dists)
    dists[idx] = np.inf
    # Sort the 19 distances
    dists_sorted = np.sort(dists)
    # Check if any distance < 0.05
    if np.any(dists_sorted < 0.05):
        return 0.0
    # Find k: first index minimizing d_{k+1} - d_k
    gaps = dists_sorted[1:] - dists_sorted[:-1]
    k = np.argmin(gaps)
    D = (dists_sorted[k] + dists_sorted[k + 1]) / 2
    if D < 0.1:
        return 0.0
    # Select four distances
    if k == 0:
        four_dists = dists_sorted[0:4]
    elif k == 17:
        four_dists = dists_sorted[15:19]
    else:
        four_dists = dists_sorted[k - 1:k + 3]
    # raw = mean of max(d/D, D/d)
    ratios = np.maximum(four_dists / D, D / four_dists)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)


def _polygon_score(polygon):
    """Compute the minimum vertex score over all vertices."""
    n = len(polygon)
    scores = []
    for i in range(n):
        scores.append(_vertex_score(polygon[i], polygon))
    return min(scores)


def _make_convex_adjusted(polygon, step_size=0.01):
    """Try to make polygon strictly convex by adjusting vertices inward/outward."""
    n = len(polygon)
    adjusted = polygon.copy()
    for _ in range(100):
        if _is_valid_convex(adjusted):
            return adjusted
        # Find vertices causing non-convexity
        for i in range(n):
            p0 = adjusted[(i - 1) % n]
            p1 = adjusted[i]
            p2 = adjusted[(i + 1) % n]
            e1 = p1 - p0
            e2 = p2 - p1
            cross = e1[0] * e2[1] - e1[1] * e2[0]
            if abs(cross) < 1e-12:
                # Move vertex slightly
                centroid = adjusted.mean(axis=0)
                direction = p1 - centroid
                norm = np.linalg.norm(direction)
                if norm > 0:
                    adjusted[i] = p1 + step_size * direction / norm
        # Small random perturbation to escape local issues
        adjusted += np.random.randn(n, 2) * 0.001
    return adjusted


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices where every vertex has approximately
    4 other vertices equidistant from it.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit with safety margin
    
    n = 20
    
    # Start with regular 20-gon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    
    # Normalize
    polygon = _normalize(polygon)
    
    best_polygon = polygon.copy()
    best_score = _polygon_score(polygon)
    
    # Try to improve by creating 4-fold equidistant structure
    # Strategy: for each vertex, try to make it equidistant to 4 specific other vertices
    # Use gradient-free optimization with convexity constraints
    
    rng = np.random.default_rng(42)
    
    # Parameterize: perturb the regular polygon slightly
    # Try different perturbation patterns that might create equidistant structure
    
    max_iterations = 50000
    for iteration in range(max_iterations):
        if time.time() > deadline - 10.0:
            break
        
        # Create a perturbed version
        # Perturb angles and radii
        angle_perturb = rng.normal(0, 0.02, n)
        radius_perturb = 1.0 + rng.normal(0, 0.02, n)
        
        new_angles = angles + angle_perturb
        new_polygon = np.stack([radius_perturb * np.cos(new_angles), 
                                 radius_perturb * np.sin(new_angles)], axis=1)
        new_polygon = _normalize(new_polygon)
        
        # Check convexity
        if not _is_valid_convex(new_polygon):
            # Try to fix
            new_polygon = _make_convex_adjusted(new_polygon)
            if not _is_valid_convex(new_polygon):
                continue
        
        score = _polygon_score(new_polygon)
        if score > best_score:
            best_score = score
            best_polygon = new_polygon.copy()
        
        # Local refinement around best
        if iteration % 100 == 0:
            # Do a local search step
            local_best = best_polygon.copy()
            for local_iter in range(50):
                if time.time() > deadline - 10.0:
                    break
                # Perturb one vertex
                vi = rng.integers(0, n)
                delta = rng.normal(0, 0.01, 2)
                candidate = local_best.copy()
                candidate[vi] += delta
                candidate = _normalize(candidate)
                if _is_valid_convex(candidate):
                    cand_score = _polygon_score(candidate)
                    if cand_score > _polygon_score(local_best):
                        local_best = candidate
            if _polygon_score(local_best) > best_score:
                best_score = _polygon_score(local_best)
                best_polygon = local_best
    
    # Final normalization
    best_polygon = _normalize(best_polygon)
    
    # Ensure validity
    if not _is_valid_convex(best_polygon):
        # Fall back to regular polygon
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        best_polygon = _normalize(best_polygon)
    
    return best_polygon


# EVOLVE-BLOCK-END
