# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_scores(polygon):
    """Compute per-vertex scores for a given polygon."""
    n = len(polygon)
    # Normalize: subtract centroid, divide by mean distance
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = dists_to_centroid.mean()
    if mean_dist <= 0.01:
        return 0.0, np.zeros(n)
    normalized = centered / mean_dist
    
    scores = np.zeros(n)
    for i in range(n):
        # Distances from vertex i to all other vertices
        diffs = normalized - normalized[i]
        d = np.linalg.norm(diffs, axis=1)
        # Remove self-distance
        d[i] = np.inf
        d_sorted = np.sort(d)
        
        # Check: if any distance < 0.05, score is 0
        if d_sorted[0] < 0.05:
            scores[i] = 0.0
            continue
        
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        D = (d_sorted[k] + d_sorted[k+1]) / 2
        if D < 0.1:
            scores[i] = 0.0
            continue
        
        # Take four distances
        if k == 0:
            four_d = d_sorted[0:4]
        elif k == 17:
            four_d = d_sorted[15:19]
        else:
            four_d = d_sorted[k-1:k+3]
        
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(four_d / D, D / four_d)
        raw = ratios.mean()
        scores[i] = min(1.0, 1.0 / raw)
    
    return scores.min(), scores

def _smooth_min_score(polygon, temperature=0.05):
    """Smooth surrogate for the min score using log-sum-exp."""
    n = len(polygon)
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = dists_to_centroid.mean()
    if mean_dist <= 0.01:
        return -1000.0
    normalized = centered / mean_dist
    
    log_scores = np.zeros(n)
    for i in range(n):
        diffs = normalized - normalized[i]
        d = np.linalg.norm(diffs, axis=1)
        d[i] = np.inf
        d_sorted = np.sort(d)
        
        if d_sorted[0] < 0.05:
            log_scores[i] = -100.0
            continue
        
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        D = (d_sorted[k] + d_sorted[k+1]) / 2
        if D < 0.1:
            log_scores[i] = -100.0
            continue
        
        if k == 0:
            four_d = d_sorted[0:4]
        elif k == 17:
            four_d = d_sorted[15:19]
        else:
            four_d = d_sorted[k-1:k+3]
        
        ratios = np.maximum(four_d / D, D / four_d)
        raw = ratios.mean()
        score = min(1.0, 1.0 / raw)
        if score <= 0:
            log_scores[i] = -100.0
        else:
            log_scores[i] = np.log(score + 1e-15)
    
    # Soft min: -log-sum-exp(-log_scores) / temperature
    max_log = np.max(log_scores)
    log_sum_exp = max_log + np.log(np.sum(np.exp((log_scores - max_log) / temperature)))
    smooth_min = -(log_sum_exp) / temperature
    return smooth_min

def _negative_smooth_min(polygon_flat, temperature=0.05):
    """Objective for minimization (negative smooth min score)."""
    n = 20
    polygon = polygon_flat.reshape(n, 2)
    return -_smooth_min_score(polygon, temperature)

def _is_valid_convex(polygon):
    """Check if polygon is strictly convex."""
    n = len(polygon)
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-10:
                return False
    
    # Check all turns are nonzero and same sign
    signs = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i+1) % n]
        p2 = polygon[(i+2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-12:
            return False
        signs.append(np.sign(cross))
    
    # All signs must be the same
    if not all(s == signs[0] for s in signs):
        return False
    
    # Total turning should be ±2π
    # For a convex polygon, sum of exterior angles = 2π
    total_turn = 0.0
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i+1) % n]
        p2 = polygon[(i+2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        turn = angle2 - angle1
        # Normalize to [-π, π]
        while turn > np.pi:
            turn -= 2 * np.pi
        while turn < -np.pi:
            turn += 2 * np.pi
        total_turn += turn
    
    if abs(abs(total_turn) - 2 * np.pi) > 0.1:
        return False
    
    return True

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with 20 vertices maximizing the minimum vertex score.
    Uses optimization with smooth surrogate for the min objective.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # 10s safety margin
    
    n = 20
    # Initial guess: regular polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x0 = np.cos(angles)
    y0 = np.sin(angles)
    x0_flat = np.concatenate([x0, y0])
    
    best_polygon = np.column_stack([x0, y0])
    best_score, _ = _compute_scores(best_polygon)
    
    # Try multiple temperatures (annealing)
    temperatures = [0.1, 0.05, 0.02]
    
    for temp in temperatures:
        if time.time() > deadline - 30:
            break
        
        # Use Nelder-Mead on the smooth objective
        try:
            result = minimize(
                _negative_smooth_min,
                x0_flat.copy(),
                args=(temp,),
                method='Nelder-Mead',
                options={
                    'maxiter': 20000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < 0:
                candidate = result.x.reshape(n, 2)
                if _is_valid_convex(candidate):
                    score, _ = _compute_scores(candidate)
                    if score > best_score:
                        best_score = score
                        best_polygon = candidate.copy()
                        x0_flat = result.x.copy()
        except Exception:
            pass
    
    # Additional local refinement with lower temperature
    if time.time() < deadline - 20:
        try:
            result = minimize(
                _negative_smooth_min,
                x0_flat.copy(),
                args=(0.01,),
                method='Nelder-Mead',
                options={
                    'maxiter': 50000,
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < 0:
                candidate = result.x.reshape(n, 2)
                if _is_valid_convex(candidate):
                    score, _ = _compute_scores(candidate)
                    if score > best_score:
                        best_score = score
                        best_polygon = candidate.copy()
        except Exception:
            pass
    
    # Final check and return
    if time.time() < deadline - 5:
        # Verify the best polygon is valid
        if _is_valid_convex(best_polygon):
            final_score, _ = _compute_scores(best_polygon)
            if final_score > 0:
                return best_polygon
    
    # Fallback: regular polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

# EVOLVE-BLOCK-END
