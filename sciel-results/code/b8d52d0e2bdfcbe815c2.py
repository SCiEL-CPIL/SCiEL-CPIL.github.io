import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import sys


def _vertex_score(distances):
    """
    distances: array of 19 distances from vertex i to others.
    Returns score for this vertex in [0, 1].
    """
    d = np.sort(distances)
    
    if np.any(d < 0.05):
        return 0.0
        
    diffs = d[1:] - d[:-1]
    k = np.argmin(diffs)
    
    D = (d[k] + d[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k-1:k+3]
        
    if len(selected) != 4:
        return 0.0

    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    
    score = min(1.0, 1.0 / raw)
    return score


def _is_valid_convex(points):
    """Check if points form a strictly convex polygon."""
    n = len(points)
    
    # Check for repeated points
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-6:
        return False
        
    # Check turns
    edges = np.roll(points, -1, axis=0) - points
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    
    if np.any(np.abs(cross) < 1e-8):
        return False
        
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return False
        
    return True


def _polygon_score(points):
    """
    points: (20, 2) array.
    Returns minimum vertex score after normalization.
    """
    centroid = np.mean(points, axis=0)
    points_norm = points - centroid
    
    dists_to_c = np.linalg.norm(points_norm, axis=1)
    mean_dist = np.mean(dists_to_c)
    
    if mean_dist <= 0.01:
        return 0.0
        
    points_norm = points_norm / mean_dist
    
    if not _is_valid_convex(points_norm):
        return 0.0
        
    n = len(points_norm)
    min_score = 1.0
    for i in range(n):
        other_indices = [j for j in range(n) if j != i]
        dists = np.linalg.norm(points_norm[other_indices] - points_norm[i], axis=1)
        s = _vertex_score(dists)
        if s < min_score:
            min_score = s
        if min_score == 0.0:
            return 0.0
            
    return min_score


def _spiral_to_points(params):
    """
    params: array of length 3 - [log_a, b, rotation]
    where a = exp(log_a), spiral is r = a * exp(b * theta), theta in [0, 2*pi]
    Returns (20, 2) array of points in cyclic order.
    """
    n = 20
    log_a, b, rotation = params
    
    # Clamp parameters to reasonable ranges
    log_a = np.clip(log_a, -1.0, 1.0)
    b = np.clip(b, -0.5, 0.5)
    rotation = np.clip(rotation, -np.pi, np.pi)
    
    a = np.exp(log_a)
    
    # Generate 20 equally spaced angles
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles = base_angles + rotation
    
    # Compute radii from spiral
    radii = a * np.exp(b * angles)
    
    # Ensure radii are positive
    radii = np.maximum(radii, 0.1)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)


def _general_to_points(params):
    """
    params: array of length 40 - first 20 are angle offsets, next 20 are radii.
    Returns (20, 2) array of points.
    """
    n = 20
    angle_offsets = params[:n]
    radii = params[n:]
    
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles = base_angles + angle_offsets
    
    radii = np.maximum(radii, 0.1)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)


def _objective_spiral(params):
    points = _spiral_to_points(params)
    score = _polygon_score(points)
    if score == 0.0:
        return 100.0
    return -score


def _objective_general(params):
    points = _general_to_points(params)
    score = _polygon_score(points)
    if score == 0.0:
        return 100.0
    return -score


def _local_search(points, max_iter=100, step_size=0.01):
    """
    Perform a local search around the given points.
    """
    n = 20
    current_points = points.copy()
    current_score = _polygon_score(current_points)
    
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            for dx in [-step_size, step_size]:
                for dy in [-step_size, step_size]:
                    trial = current_points.copy()
                    trial[i] += [dx, dy]
                    trial_score = _polygon_score(trial)
                    if trial_score > current_score:
                        current_points = trial
                        current_score = trial_score
                        improved = True
        if not improved:
            break
        step_size *= 0.9
    
    return current_points, current_score


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that maximizes the minimum vertex score.
    
    Uses a two-phase approach:
    1. Global search using differential evolution on both spiral and general parameterizations
    2. Local refinement using Nelder-Mead and coordinate descent
    
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s limit
    
    n = 20
    
    # Baseline: regular polygon
    best_points = _spiral_to_points(np.array([0.0, 0.0, 0.0]))
    best_score = _polygon_score(best_points)
    
    # Phase 1: Global search using Differential Evolution
    # Try spiral parameterization first (3 params - much smaller search space)
    
    # Spiral bounds: [log_a, b, rotation]
    spiral_bounds = [(-1.0, 1.0), (-0.5, 0.5), (-np.pi, np.pi)]
    
    try:
        res_de_spiral = differential_evolution(
            _objective_spiral, 
            spiral_bounds, 
            maxiter=30, 
            popsize=8,
            tol=1e-4,
            seed=42,
            polish=False
        )
        
        if res_de_spiral.fun < 100.0:
            current_score = -res_de_spiral.fun
            if current_score > best_score:
                best_score = current_score
                best_points = _spiral_to_points(res_de_spiral.x)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 120.0:
        return best_points
    
    # Try general parameterization (40 params)
    angle_bound = 0.4 * (2 * np.pi / n)
    radius_bounds = (0.5, 1.5)
    general_bounds = [(-angle_bound, angle_bound)] * n + [radius_bounds] * n
    
    try:
        res_de_general = differential_evolution(
            _objective_general, 
            general_bounds, 
            maxiter=20, 
            popsize=6,
            tol=1e-4,
            seed=123,
            polish=False
        )
        
        if res_de_general.fun < 100.0:
            current_score = -res_de_general.fun
            if current_score > best_score:
                best_score = current_score
                best_points = _general_to_points(res_de_general.x)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 60.0:
        return best_points
    
    # Phase 2: Local refinement using Nelder-Mead
    # Convert best points to general parameterization for refinement
    
    # Extract current angles and radii from best_points
    centroid = np.mean(best_points, axis=0)
    points_centered = best_points - centroid
    current_radii = np.linalg.norm(points_centered, axis=1)
    current_angles = np.arctan2(points_centered[:, 1], points_centered[:, 0])
    
    # Unwrap angles to be in [0, 2*pi)
    current_angles = np.mod(current_angles, 2 * np.pi)
    
    # Sort by angle to get proper cyclic order
    sort_idx = np.argsort(current_angles)
    current_radii = current_radii[sort_idx]
    current_angles = current_angles[sort_idx]
    
    # Compute angle offsets from base
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    current_angle_offsets = current_angles - base_angles
    
    # Handle wrap-around for angle offsets
    for i in range(n):
        if current_angle_offsets[i] > np.pi:
            current_angle_offsets[i] -= 2 * np.pi
        elif current_angle_offsets[i] < -np.pi:
            current_angle_offsets[i] += 2 * np.pi
    
    current_params = np.concatenate([current_angle_offsets, current_radii])
    
    # Run Nelder-Mead refinement
    max_refinements = 3
    for i in range(max_refinements):
        if time.time() > deadline - 10.0:
            break
            
        if i == 0:
            refine_params = current_params.copy()
        else:
            refine_params = current_params + 0.01 * np.random.randn(2*n)
            
        try:
            res_nm = minimize(
                _objective_general, 
                refine_params, 
                method='Nelder-Mead', 
                options={'xatol': 1e-6, 'fatol': 1e-7, 'maxiter': 1500, 'maxfev': 4000, 'adaptive': True}
            )
            
            if res_nm.fun < 100.0:
                nm_score = -res_nm.fun
                if nm_score > best_score:
                    best_score = nm_score
                    best_points = _general_to_points(res_nm.x)
                    # Update current_params for next iteration
                    centroid = np.mean(best_points, axis=0)
                    points_centered = best_points - centroid
                    current_radii = np.linalg.norm(points_centered, axis=1)
                    current_angles = np.arctan2(points_centered[:, 1], points_centered[:, 0])
                    current_angles = np.mod(current_angles, 2 * np.pi)
                    sort_idx = np.argsort(current_angles)
                    current_radii = current_radii[sort_idx]
                    current_angles = current_angles[sort_idx]
                    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
                    current_angle_offsets = current_angles - base_angles
                    for j in range(n):
                        if current_angle_offsets[j] > np.pi:
                            current_angle_offsets[j] -= 2 * np.pi
                        elif current_angle_offsets[j] < -np.pi:
                            current_angle_offsets[j] += 2 * np.pi
                    current_params = np.concatenate([current_angle_offsets, current_radii])
        except Exception:
            continue
    
    # Phase 3: Final local search to polish
    if time.time() < deadline - 15.0:
        polished_points, polished_score = _local_search(best_points, max_iter=50, step_size=0.01)
        if polished_score > best_score:
            best_points = polished_points
            best_score = polished_score
    
    return best_points


if __name__ == "__main__":
    # Test
    pts = equidistant_points_polygon20()
    print("Score:", _polygon_score(pts))
