# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score_vertex(v, pts):
    """
    Compute the vertex score for a single vertex v given all points pts (20x2).
    Follows the task contract exactly.
    """
    # Distances from v to all other points
    d = np.linalg.norm(pts - v, axis=1)
    # Remove self (distance 0)
    d = np.delete(d, np.argmin(d))
    
    if len(d) != 19:
        return 0.0
    
    d_sorted = np.sort(d)
    
    # Check if any distance < 0.05
    if d_sorted[0] < 0.05:
        return 0.0
    
    # Find k: first index minimizing d_{k+1} - d_k
    diffs = np.diff(d_sorted)
    k = np.argmin(diffs)
    
    D = (d_sorted[k] + d_sorted[k + 1]) / 2.0
    
    if D < 0.1:
        return 0.0
    
    # Select four distances
    if k == 0:
        four_d = d_sorted[0:4]
    elif k == 17:
        four_d = d_sorted[15:19]
    else:
        four_d = d_sorted[k - 1:k + 3]
    
    # raw = mean of max(d/D, D/d)
    ratios = np.maximum(four_d / D, D / four_d)
    raw = np.mean(ratios)
    
    if raw <= 0:
        return 0.0
    
    return min(1.0, 1.0 / raw)


def _score_polygon(pts):
    """
    Compute the minimum vertex score over all vertices.
    """
    n = len(pts)
    scores = np.array([_score_vertex(pts[i], pts) for i in range(n)])
    return np.min(scores)


def _is_strictly_convex(pts):
    """
    Check if the polygon is strictly convex.
    """
    n = len(pts)
    
    # Check no repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                return False
    
    # Check all turns are nonzero and of one sign
    signs = set()
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        
        v1 = p1 - p0
        v2 = p2 - p1
        
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        if abs(cross) < 1e-12:
            return False
        
        signs.add(np.sign(cross))
    
    if len(signs) != 1:
        return False
    
    return True


def _normalize(pts):
    """
    Normalize: subtract centroid, divide by mean distance to centroid.
    """
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    
    dists = np.linalg.norm(pts_centered, axis=1)
    mean_dist = np.mean(dists)
    
    if mean_dist <= 0.01:
        return None
    
    return pts_centered / mean_dist


def _evaluate(pts):
    """
    Evaluate the polygon: normalize, check convexity, compute score.
    Returns (score, normalized_pts) or (0, None) if invalid.
    """
    norm_pts = _normalize(pts)
    if norm_pts is None:
        return 0.0, None
    
    if not _is_strictly_convex(norm_pts):
        return 0.0, None
    
    score = _score_polygon(norm_pts)
    return score, norm_pts


def _neg_score(params):
    """
    Objective function for optimization: negative of the minimum vertex score.
    params: 40-dimensional vector representing 20 (x,y) points.
    """
    pts = params.reshape(20, 2)
    score, _ = _evaluate(pts)
    return -score


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with 20 vertices optimized for the vertex score metric.
    
    Returns:
        polygon: np.ndarray of shape (20, 2) with vertex coordinates in cyclic order.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget with safety margin
    
    n = 20
    
    # Start with regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score, _ = _evaluate(best_pts)
    
    # Multi-start optimization
    n_starts = 50
    rng = np.random.default_rng(42)
    
    for start_idx in range(n_starts):
        if time.time() > deadline - 60:
            break
        
        # Generate initial guess: regular polygon with some perturbation
        if start_idx == 0:
            # Regular polygon
            init = best_pts.copy()
        else:
            # Perturbed regular polygon
            init = best_pts + rng.normal(0, 0.1, size=(n, 2))
        
        # Try L-BFGS-B optimization
        try:
            result = minimize(
                _neg_score,
                init.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 5000}
            )
            
            if time.time() > deadline - 30:
                break
            
            if result.fun < -best_score:
                new_pts = result.x.reshape(n, 2)
                new_score, _ = _evaluate(new_pts)
                if new_score > best_score:
                    best_pts = new_pts
                    best_score = new_score
                    
        except Exception:
            continue
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 30:
        try:
            result = minimize(
                _neg_score,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.fun < -best_score:
                new_pts = result.x.reshape(n, 2)
                new_score, _ = _evaluate(new_pts)
                if new_score > best_score:
                    best_pts = new_pts
                    best_score = new_score
                    
        except Exception:
            pass
    
    return best_pts


# EVOLVE-BLOCK-END
