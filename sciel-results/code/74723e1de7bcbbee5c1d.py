# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _is_strictly_convex(points: np.ndarray) -> bool:
    """Check if the polygon defined by points is strictly convex."""
    n = len(points)
    if n < 3:
        return False
    
    # Check for repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-8:
                return False

    # Check cross products for consistent sign (convexity)
    # For a CCW convex polygon, cross product of consecutive edges should be positive
    cross_products = np.zeros(n)
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross_products[i] = v1[0] * v2[1] - v1[1] * v2[0]
    
    # All cross products must be non-zero and of the same sign
    if np.any(np.abs(cross_products) < 1e-12):
        return False
        
    if np.all(cross_products > 0) or np.all(cross_products < 0):
        return True
    else:
        return False

def _score_polygon(polygon: np.ndarray) -> float:
    """Compute the minimum vertex score for a given polygon."""
    n = len(polygon)
    if n != 20:
        return 0.0
        
    # Check convexity first to avoid wasting time on invalid shapes
    if not _is_strictly_convex(polygon):
        return 0.0

    # Normalize: subtract centroid, divide by mean distance
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt((centered ** 2).sum(axis=1))
    mean_dist = dists_to_centroid.mean()
    if mean_dist <= 0.01:
        return 0.0
    normalized = centered / mean_dist
    
    min_vertex_score = 1.0
    for i in range(n):
        # Distances from vertex i to all other vertices
        diffs = normalized - normalized[i]
        d = np.sqrt((diffs ** 2).sum(axis=1))
        # Remove self-distance
        d = np.delete(d, i)
        d = np.sort(d)
        
        # Check if any distance < 0.05
        if d[0] < 0.05:
            return 0.0
        
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = np.diff(d)
        k = np.argmin(gaps)
        D = (d[k] + d[k + 1]) / 2
        if D < 0.1:
            return 0.0
        
        # Select four distances
        if k == 0:
            selected = d[0:4]
        elif k == 17:  # k=17 means d[17] and d[18] are the closest gap
            selected = d[15:19]
        else:
            selected = d[k-1:k+3]
        
        # raw = mean over these four d of max(d/D, D/d)
        ratios = np.maximum(selected / D, D / selected)
        raw = ratios.mean()
        vertex_score = min(1.0, 1.0 / raw)
        min_vertex_score = min(min_vertex_score, vertex_score)
    
    return min_vertex_score


def _smooth_vertex_scores(normalized: np.ndarray, beta: float = 50.0) -> np.ndarray:
    """Compute smooth surrogate vertex scores using soft-min for the min over vertices.
    
    Returns array of per-vertex scores (not yet minimized), plus we apply soft-min
    across vertices for the optimization objective.
    """
    n = len(normalized)
    vertex_scores = np.ones(n)
    
    for i in range(n):
        diffs = normalized - normalized[i]
        d = np.sqrt((diffs ** 2).sum(axis=1))
        d = np.delete(d, i)
        d = np.sort(d)
        
        # Smooth penalty for d[0] < 0.05
        if d[0] < 0.05:
            # Use a smooth sigmoid penalty
            penalty = np.exp(-beta * (0.05 - d[0]))
            vertex_scores[i] *= penalty
        
        gaps = np.diff(d)
        k = np.argmin(gaps)
        D = (d[k] + d[k + 1]) / 2
        
        # Smooth penalty for D < 0.1
        if D < 0.1:
            penalty = np.exp(-beta * (0.1 - D))
            vertex_scores[i] *= penalty
        
        if k == 0:
            selected = d[0:4]
        elif k == 17:
            selected = d[15:19]
        else:
            selected = d[k-1:k+3]
        
        # Smooth max(d/D, D/d) using log-sum-exp trick
        # max(a, b) ≈ (1/beta) * log(exp(beta*a) + exp(beta*b)) for large beta
        a = selected / D
        b = D / selected
        max_val = np.maximum(a, b)
        # For smoothness, use a soft max
        soft_max = (1.0 / beta) * np.log(np.exp(beta * a) + np.exp(beta * b))
        # But this can overflow; use the stable version
        m = np.maximum(a, b)
        soft_max = m + (1.0 / beta) * np.log(np.exp(beta * (a - m)) + np.exp(beta * (b - m)))
        
        raw = soft_max.mean()
        # Smooth min(1, 1/raw)
        # Use: min(1, 1/raw) ≈ 1 / (1 + softplus(raw - 1)) approximately
        # Or more simply: for raw close to 1, use a smooth cap
        if raw <= 1.0:
            vertex_scores[i] = 1.0
        else:
            # Smooth transition: 1/raw but smoothed near raw=1
            # Use 1/(raw + epsilon) with a smooth cap
            vertex_scores[i] = 1.0 / (raw + 1e-10)
    
    return vertex_scores


def _neg_score_smooth(params: np.ndarray, beta: float = 30.0) -> float:
    """Negative smooth score for minimization using soft-min across vertices."""
    polygon = params.reshape(20, 2)
    
    if not np.all(np.isfinite(polygon)):
        return 1000.0
        
    # Quick convexity check
    n = 20
    cross_products = np.zeros(n)
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross_products[i] = v1[0] * v2[1] - v1[1] * v2[0]
    
    if np.any(np.abs(cross_products) < 1e-12):
        return 100.0 + np.sum(np.abs(cross_products[cross_products <= 0])) * 10.0
    if not (np.all(cross_products > 0) or np.all(cross_products < 0)):
        return 100.0 + np.sum(np.abs(cross_products[cross_products <= 0])) * 10.0
    
    # Normalize
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt((centered ** 2).sum(axis=1))
    mean_dist = dists_to_centroid.mean()
    if mean_dist <= 0.01:
        return 1000.0
    normalized = centered / mean_dist
    
    # Get smooth vertex scores
    v_scores = _smooth_vertex_scores(normalized, beta)
    
    # Soft-min across vertices: -log(sum(exp(-beta * s_i))) / beta
    # This approximates min(s_i) for large beta
    min_s = np.min(v_scores)
    # Use soft min for smoothness
    shifted = -beta * v_scores
    max_shifted = np.max(shifted)
    soft_min = -(max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))) / beta
    
    return -soft_min


def _neg_score_smooth(params: np.ndarray, beta: float = 20.0) -> float:
    """Negative smooth score for minimization using soft-min across vertices.
    
    Uses a smooth surrogate that avoids discontinuities from k-index changes
    and the min(1, 1/raw) cap, allowing the optimizer to navigate better.
    """
    polygon = params.reshape(20, 2)
    
    # Check for finite values
    if not np.all(np.isfinite(polygon)):
        return 1000.0
        
    # Quick convexity check with penalty
    n = 20
    cross_products = np.zeros(n)
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross_products[i] = v1[0] * v2[1] - v1[1] * v2[0]
    
    if np.any(np.abs(cross_products) < 1e-10):
        return 100.0 + np.sum(np.abs(cross_products[cross_products <= 0])) * 10.0
    if not (np.all(cross_products > 0) or np.all(cross_products < 0)):
        return 100.0 + np.sum(np.abs(cross_products[cross_products <= 0])) * 10.0

    # Normalize
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt((centered ** 2).sum(axis=1))
    mean_dist = dists_to_centroid.mean()
    if mean_dist <= 0.01:
        return 1000.0
    normalized = centered / mean_dist
    
    vertex_scores = np.ones(n)
    for i in range(n):
        diffs = normalized - normalized[i]
        d = np.sqrt((diffs ** 2).sum(axis=1))
        d = np.delete(d, i)
        d = np.sort(d)
        
        # Smooth penalty for d[0] < 0.05
        if d[0] < 0.05:
            vertex_scores[i] *= np.exp(-beta * (0.05 - d[0]))
        
        gaps = np.diff(d)
        k = np.argmin(gaps)
        D = (d[k] + d[k + 1]) / 2
        
        # Smooth penalty for D < 0.1
        if D < 0.1:
            vertex_scores[i] *= np.exp(-beta * (0.1 - D))
        
        if k == 0:
            selected = d[0:4]
        elif k == 17:
            selected = d[15:19]
        else:
            selected = d[k-1:k+3]
        
        # Smooth max(d/D, D/d)
        a = selected / D
        b = D / selected
        m = np.maximum(a, b)
        soft_max = m + (1.0 / beta) * np.log(np.exp(beta * (a - m)) + np.exp(beta * (b - m)))
        raw = soft_max.mean()
        
        # Smooth min(1, 1/raw): use 1/(1+softplus(raw-1)) approximation
        # Simpler: if raw > 1, use 1/raw; else 1.0, but smooth the transition
        if raw > 1.0:
            # Smooth cap: transition from 1.0 to 1/raw over a small range
            x = raw - 1.0
            smooth_cap = 1.0 / (1.0 + x * (1.0 - 1.0/(raw))) if raw > 1.0 else 1.0
            # Simpler smooth: 1/(raw + 1e-6)
            vertex_scores[i] = min(1.0, 1.0 / (raw + 1e-8))
        else:
            vertex_scores[i] = 1.0
    
    # Soft-min across vertices for smoothness
    min_s = np.min(vertex_scores)
    shifted = -beta * vertex_scores
    max_shifted = np.max(shifted)
    soft_min = -(max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))) / beta
    
    return -soft_min


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Uses direct coordinate optimization starting from a regular polygon.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Initial guess: Regular 20-gon
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    regular_polygon = np.stack([x, y], axis=1)
    
    # Verify regular polygon is valid
    reg_score = _score_polygon(regular_polygon)
    
    best_score = reg_score
    best_polygon = regular_polygon
    
    # Try to improve using optimization
    initial_params = regular_polygon.flatten()
    
    try:
        # Use Nelder-Mead with smooth surrogate for better navigation
        result = minimize(_neg_score_smooth, initial_params, method='Nelder-Mead',
                         options={'maxiter': 8000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
        
        if result.fun < -best_score * 0.95:  # Allow some tolerance
            candidate = result.x.reshape(20, 2)
            # Final validation with exact objective
            if _is_strictly_convex(candidate) and np.all(np.isfinite(candidate)):
                cand_score = _score_polygon(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_polygon = candidate
    except Exception:
        pass
    
    # If time allows, try a few random perturbations of the regular polygon
    if time.time() < deadline - 100:
        np.random.seed(42)
        for _ in range(20):
            if time.time() > deadline - 50:
                break
            # Small random perturbation
            noise = np.random.normal(0, 0.05, (20, 2))
            perturbed = regular_polygon + noise
            # Ensure it's still roughly convex by checking
            if _is_strictly_convex(perturbed):
                p_score = _score_polygon(perturbed)
                if p_score > best_score:
                    best_score = p_score
                    best_polygon = perturbed
                    # Try to optimize from this new point using smooth objective
                    try:
                        res2 = minimize(_neg_score_smooth, perturbed.flatten(), method='Nelder-Mead',
                                     options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9})
                        if res2.fun < -best_score * 0.95:
                            cand2 = res2.x.reshape(20, 2)
                            if _is_strictly_convex(cand2):
                                s2 = _score_polygon(cand2)
                                if s2 > best_score:
                                    best_score = s2
                                    best_polygon = cand2
                    except Exception:
                        pass

    return best_polygon


# EVOLVE-BLOCK-END
