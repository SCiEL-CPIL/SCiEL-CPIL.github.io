# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _normalize(polygon):
    """Subtract centroid and divide by mean distance to centroid."""
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = dists.mean()
    if mean_dist <= 0.01:
        return centered, mean_dist
    return centered / mean_dist, mean_dist

def _check_strictly_convex(polygon):
    """Check if polygon is strictly convex."""
    n = len(polygon)
    p0 = polygon
    p1 = np.roll(polygon, -1, axis=0)
    p2 = np.roll(polygon, -2, axis=0)
    v1 = p1 - p0
    v2 = p2 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    if np.any(np.abs(cross) < 1e-9):
        return False
    signs = np.sign(cross)
    if not np.all(signs == signs[0]):
        return False
    # Check total turning
    angles = []
    for i in range(n):
        a1 = np.arctan2(v1[i, 1], v1[i, 0])
        a2 = np.arctan2(v2[i, 1], v2[i, 0])
        da = a2 - a1
        while da > np.pi:
            da -= 2 * np.pi
        while da < -np.pi:
            da += 2 * np.pi
        angles.append(da)
    total_turn = sum(angles)
    if abs(abs(total_turn) - 2 * np.pi) > 1e-4:
        return False
    return True

def _vertex_score(v_idx, polygon):
    """Compute the vertex score for vertex at v_idx in polygon."""
    n = len(polygon)
    v = polygon[v_idx]
    diffs = polygon - v
    dists = np.linalg.norm(diffs, axis=1)
    dists[v_idx] = np.inf
    other_dists = dists[~np.isinf(dists)]
    sorted_dists = np.sort(other_dists)
    
    if sorted_dists[0] < 0.05:
        return 0.0
        
    gaps = np.diff(sorted_dists)
    k = np.argmin(gaps)
    D = (sorted_dists[k] + sorted_dists[k + 1]) / 2
    if D < 0.1:
        return 0.0
        
    if k == 0:
        selected = sorted_dists[0:4]
    elif k == 17:
        selected = sorted_dists[15:19]
    else:
        selected = sorted_dists[k - 1:k + 3]
        
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _score_polygon(polygon):
    """Compute the minimum vertex score over all vertices."""
    polygon, mean_dist = _normalize(polygon)
    if not _check_strictly_convex(polygon):
        return 0.0
    scores = np.array([_vertex_score(i, polygon) for i in range(len(polygon))])
    return float(np.min(scores))

def _angle_to_polygon(angles, n=20):
    """Convert angle deviations to a polygon."""
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    final_angles = base_angles + angles
    final_angles = final_angles - np.mean(final_angles)
    final_angles = np.sort(final_angles)
    final_angles = final_angles % (2 * np.pi)
    final_angles = np.sort(final_angles)
    return np.stack([np.cos(final_angles), np.sin(final_angles)], axis=1)

def _soft_min(values, beta=50.0):
    """Compute soft minimum of values using log-sum-exp."""
    min_val = np.min(values)
    shifted = values - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_val - np.log(np.sum(np.exp(-beta * shifted))) / beta

def _score_polygon_smooth(polygon, beta=50.0):
    """Compute smooth approximation of minimum vertex score."""
    polygon, mean_dist = _normalize(polygon)
    if not _check_strictly_convex(polygon):
        return 0.0
    scores = np.array([_vertex_score(i, polygon) for i in range(len(polygon))])
    return float(_soft_min(scores, beta))

def _objective_smooth(angles, n=20, beta=50.0):
    """Negative smooth score for minimization."""
    polygon = _angle_to_polygon(angles, n)
    score = _score_polygon_smooth(polygon, beta)
    return -score

def _objective(angles, n=20):
    """Negative score for minimization."""
    polygon = _angle_to_polygon(angles, n)
    score = _score_polygon(polygon)
    return -score

def _optimize_polygon(start_time, max_time=600.0):
    n = 20
    deadline = start_time + max_time - 30.0
    
    # Start with regular polygon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    best_score = _score_polygon(best_polygon)
    
    np.random.seed(42)
    
    # Phase 1: Multi-restart with L-BFGS-B using smooth objective
    noise_scales = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    for noise_scale in noise_scales:
        if time.time() > deadline:
            break
        
        perturbed = np.random.randn(n) * noise_scale
        start_angles = perturbed
        
        # First use smooth objective with L-BFGS-B for faster convergence
        result = minimize(
            _objective_smooth,
            start_angles,
            args=(n, 50.0),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if result.success or result.fun < 0:
            final_angles = result.x
            final_polygon = _angle_to_polygon(final_angles, n)
            final_score = _score_polygon(final_polygon)
            
            if final_score > best_score:
                best_polygon = final_polygon
                best_score = final_score
    
    # Phase 2: Nelder-Mead refinement on exact objective
    if time.time() < deadline:
        current_angles = np.arctan2(best_polygon[:, 1], best_polygon[:, 0])
        base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        current_angles = current_angles - base_angles
        current_angles = np.unwrap(current_angles)
        
        result = minimize(
            _objective,
            current_angles,
            args=(n,),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        if result.success or result.fun < 0:
            final_angles = result.x
            final_polygon = _angle_to_polygon(final_angles, n)
            final_score = _score_polygon(final_polygon)
            
            if final_score > best_score:
                best_polygon = final_polygon
                best_score = final_score
    
    # Phase 3: High-precision coordinate ascent polish
    if time.time() < deadline:
        current_angles = np.arctan2(best_polygon[:, 1], best_polygon[:, 0])
        base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        current_angles = current_angles - base_angles
        current_angles = np.unwrap(current_angles)
        
        step_size = 0.01
        for iteration in range(2000):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for delta in [step_size, -step_size]:
                    trial_angles = current_angles.copy()
                    trial_angles[i] += delta
                    trial_polygon = _angle_to_polygon(trial_angles, n)
                    trial_score = _score_polygon(trial_polygon)
                    if trial_score > best_score + 1e-10:
                        current_angles = trial_angles
                        best_polygon = trial_polygon
                        best_score = trial_score
                        improved = True
                        break
            if not improved:
                step_size *= 0.8
                if step_size < 1e-9:
                    break
    
    # Phase 4: Final L-BFGS-B polish with very tight tolerance
    if time.time() < deadline:
        current_angles = np.arctan2(best_polygon[:, 1], best_polygon[:, 0])
        base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        current_angles = current_angles - base_angles
        current_angles = np.unwrap(current_angles)
        
        result = minimize(
            _objective_smooth,
            current_angles,
            args=(n, 100.0),
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.success or result.fun < 0:
            final_angles = result.x
            final_polygon = _angle_to_polygon(final_angles, n)
            final_score = _score_polygon(final_polygon)
            
            if final_score > best_score:
                best_polygon = final_polygon
                best_score = final_score
    
    return best_polygon, best_score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    polygon, score = _optimize_polygon(start_time)
    return polygon

# EVOLVE-BLOCK-END
