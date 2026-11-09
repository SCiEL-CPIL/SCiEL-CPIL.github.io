# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time


def _compute_vertex_scores(polygon: np.ndarray) -> np.ndarray:
    """
    Compute the per-vertex equidistance score for each vertex of a convex polygon.
    
    Parameters:
        polygon: (20, 2) array of vertex coordinates in cyclic order
        
    Returns:
        (20,) array of vertex scores in [0, 1]
    """
    n = len(polygon)
    scores = np.zeros(n)
    
    for i in range(n):
        v = polygon[i]
        # Compute distances from v to all other vertices
        diffs = polygon - v
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        # Exclude self-distance
        dists = np.delete(dists, i)
        # Sort distances
        dists = np.sort(dists)
        
        # Check for too-close points
        if any(d < 0.05 for d in dists):
            scores[i] = 0.0
            continue
        
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = np.diff(dists)
        k = int(np.argmin(gaps))
        
        D = (dists[k] + dists[k + 1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
        
        # Select four distances
        if k == 0:
            selected = dists[0:4]
        elif k == 17:
            selected = dists[15:19]
        else:
            selected = dists[k-1:k+3]
        
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        
        scores[i] = min(1.0, 1.0 / raw)
    
    return scores


def _normalize_polygon(polygon: np.ndarray) -> np.ndarray:
    """
    Normalize polygon: subtract centroid, divide by mean distance to centroid.
    
    Parameters:
        polygon: (n, 2) array of vertex coordinates
        
    Returns:
        Normalized polygon, or None if invalid
    """
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists = np.sqrt(np.sum(centered**2, axis=1))
    mean_dist = np.mean(dists)
    
    if mean_dist <= 0.01:
        return None
    
    normalized = centered / mean_dist
    return normalized


def _is_strictly_convex(polygon: np.ndarray) -> bool:
    """
    Check if polygon is strictly convex with consistent turning direction.
    
    Parameters:
        polygon: (n, 2) array of vertex coordinates in cyclic order
        
    Returns:
        True if strictly convex, False otherwise
    """
    n = len(polygon)
    
    # Check for repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-10:
                return False
    
    # Compute cross products of consecutive edges
    signs = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        edge1 = p1 - p0
        edge2 = p2 - p1
        cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
        if abs(cross) < 1e-10:
            return False
        signs.append(1 if cross > 0 else -1)
    
    # All turns must be same sign
    if len(set(signs)) != 1:
        return False
    
    return True


def _objective(params: np.ndarray) -> float:
    """
    Objective function for optimization: minimize negative of minimum vertex score.
    
    Parameters:
        params: array of parameters to optimize
        
    Returns:
        Negative of the minimum vertex score (to be minimized)
    """
    n = 20
    
    # Interpret params based on strategy
    # Strategy: use radial deviations from regular polygon
    # params[0:n] = radial deviations, params[n] = global rotation
    radial_devs = params[:n]
    rotation = params[n]
    
    # Base angles for regular 20-gon
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + rotation
    
    # Radii: 1 + radial deviations
    radii = 1.0 + radial_devs
    
    # Ensure positive radii
    radii = np.maximum(radii, 0.1)
    
    # Compute polygon
    x = radii * np.cos(base_angles)
    y = radii * np.sin(base_angles)
    polygon = np.stack([x, y], axis=1)
    
    # Normalize
    normalized = _normalize_polygon(polygon)
    if normalized is None:
        return 1.0  # Bad score
    
    # Check convexity
    if not _is_strictly_convex(normalized):
        return 1.0  # Bad score
    
    # Compute scores
    scores = _compute_vertex_scores(normalized)
    min_score = np.min(scores)
    
    # Return negative for minimization
    return -min_score


def _objective_radial_only(params: np.ndarray) -> float:
    """
    Alternative objective: optimize only radial deviations (no rotation).
    """
    n = 20
    radial_devs = params
    
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = 1.0 + radial_devs
    radii = np.maximum(radii, 0.1)
    
    x = radii * np.cos(base_angles)
    y = radii * np.sin(base_angles)
    polygon = np.stack([x, y], axis=1)
    
    normalized = _normalize_polygon(polygon)
    if normalized is None:
        return 1.0
    
    if not _is_strictly_convex(normalized):
        return 1.0
    
    scores = _compute_vertex_scores(normalized)
    return -np.min(scores)


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to maximize the minimum
    vertex equidistance score.
    
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second budget with safety margin
    
    n = 20
    
    # Start with regular polygon as baseline
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    baseline_polygon = np.stack([np.cos(base_angles), np.sin(base_angles)], axis=1)
    baseline_normalized = _normalize_polygon(baseline_polygon)
    baseline_scores = _compute_vertex_scores(baseline_normalized)
    best_score = np.min(baseline_scores)
    best_polygon = baseline_polygon.copy()
    
    # Strategy 1: Differential Evolution on radial deviations
    try:
        if time.time() < deadline - 60.0:
            # Define bounds for radial deviations
            bounds = [(-0.3, 0.3)] * n
            
            # Run differential evolution
            result_de = differential_evolution(
                _objective_radial_only,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False
            )
            
            if result_de.fun < -best_score:
                radial_devs = result_de.x
                base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
                radii = 1.0 + radial_devs
                radii = np.maximum(radii, 0.1)
                x = radii * np.cos(base_angles)
                y = radii * np.sin(base_angles)
                polygon = np.stack([x, y], axis=1)
                normalized = _normalize_polygon(polygon)
                if normalized is not None and _is_strictly_convex(normalized):
                    scores = _compute_vertex_scores(normalized)
                    if np.min(scores) > best_score:
                        best_score = np.min(scores)
                        best_polygon = polygon.copy()
    except Exception:
        pass
    
    # Strategy 2: Nelder-Mead refinement from best DE result
    try:
        if time.time() < deadline - 30.0:
            # Get current best radial deviations
            if best_polygon is not None:
                # Extract radial deviations from best_polygon
                centroid = np.mean(best_polygon, axis=0)
                centered = best_polygon - centroid
                angles = np.arctan2(centered[:, 1], centered[:, 0])
                # Sort by angle to maintain cyclic order
                sort_idx = np.argsort(angles)
                sorted_angles = angles[sort_idx]
                sorted_radii = np.sqrt(np.sum(centered[sort_idx]**2, axis=1))
                mean_radius = np.mean(sorted_radii)
                radial_devs = sorted_radii / mean_radius - 1.0
                
                # Nelder-Mead refinement
                result_nm = minimize(
                    _objective_radial_only,
                    radial_devs,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                
                if result_nm.fun < -best_score:
                    radial_devs = result_nm.x
                    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
                    radii = 1.0 + radial_devs
                    radii = np.maximum(radii, 0.1)
                    x = radii * np.cos(base_angles)
                    y = radii * np.sin(base_angles)
                    polygon = np.stack([x, y], axis=1)
                    normalized = _normalize_polygon(polygon)
                    if normalized is not None and _is_strictly_convex(normalized):
                        scores = _compute_vertex_scores(normalized)
                        if np.min(scores) > best_score:
                            best_score = np.min(scores)
                            best_polygon = polygon.copy()
    except Exception:
        pass
    
    # Strategy 3: Try with rotation parameter
    try:
        if time.time() < deadline - 10.0:
            # Add rotation to optimization
            def _objective_with_rotation(params):
                radial_devs = params[:n]
                rotation = params[n]
                
                base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + rotation
                radii = 1.0 + radial_devs
                radii = np.maximum(radii, 0.1)
                
                x = radii * np.cos(base_angles)
                y = radii * np.sin(base_angles)
                polygon = np.stack([x, y], axis=1)
                
                normalized = _normalize_polygon(polygon)
                if normalized is None:
                    return 1.0
                
                if not _is_strictly_convex(normalized):
                    return 1.0
                
                scores = _compute_vertex_scores(normalized)
                return -np.min(scores)
            
            # Start from current best
            if best_polygon is not None:
                centroid = np.mean(best_polygon, axis=0)
                centered = best_polygon - centroid
                angles = np.arctan2(centered[:, 1], centered[:, 0])
                sort_idx = np.argsort(angles)
                sorted_angles = angles[sort_idx]
                sorted_radii = np.sqrt(np.sum(centered[sort_idx]**2, axis=1))
                mean_radius = np.mean(sorted_radii)
                radial_devs = sorted_radii / mean_radius - 1.0
                rotation = 0.0
                
                initial_params = np.concatenate([radial_devs, [rotation]])
                bounds = [(-0.3, 0.3)] * n + [(-np.pi, np.pi)]
                
                result_rot = differential_evolution(
                    _objective_with_rotation,
                    bounds,
                    maxiter=50,
                    popsize=10,
                    tol=1e-4,
                    seed=42,
                    polish=False,
                    x0=initial_params
                )
                
                if result_rot.fun < -best_score:
                    radial_devs = result_rot.x[:n]
                    rotation = result_rot.x[n]
                    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + rotation
                    radii = 1.0 + radial_devs
                    radii = np.maximum(radii, 0.1)
                    x = radii * np.cos(base_angles)
                    y = radii * np.sin(base_angles)
                    polygon = np.stack([x, y], axis=1)
                    normalized = _normalize_polygon(polygon)
                    if normalized is not None and _is_strictly_convex(normalized):
                        scores = _compute_vertex_scores(normalized)
                        if np.min(scores) > best_score:
                            best_score = np.min(scores)
                            best_polygon = polygon.copy()
    except Exception:
        pass
    
    # Final check: ensure we return a valid polygon
    final_normalized = _normalize_polygon(best_polygon)
    if final_normalized is None or not _is_strictly_convex(final_normalized):
        # Fall back to regular polygon
        return baseline_polygon
    
    return best_polygon


# EVOLVE-BLOCK-END
