# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _normalize(polygon):
    """Normalize: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return None
    return centered / mean_dist

def _check_convex(polygon):
    """Check if polygon is strictly convex with consistent turning."""
    n = len(polygon)
    if n < 3:
        return False
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-10:
                return False
    # Compute cross products of consecutive edges
    cross_signs = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i+1) % n]
        p2 = polygon[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0]*e2[1] - e1[1]*e2[0]
        if abs(cross) < 1e-10:
            return False
        cross_signs.append(np.sign(cross))
    # All must be same sign
    if not all(s == cross_signs[0] for s in cross_signs):
        return False
    return True

def _vertex_score(polygon, v_idx):
    """Compute the vertex score for vertex v_idx."""
    n = len(polygon)
    v = polygon[v_idx]
    dists = np.linalg.norm(polygon - v, axis=1)
    dists[v_idx] = np.inf  # exclude self
    dists = np.sort(dists)
    
    # Check if any distance < 0.05
    if dists[0] < 0.05:
        return 0.0
    
    # Find first index minimizing d_{k+1} - d_k
    gaps = np.diff(dists)
    k = np.argmin(gaps)
    D = (dists[k] + dists[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
    
    # Select 4 distances
    if k == 0:
        selected = dists[0:4]
    elif k == 17:
        selected = dists[15:19]
    else:
        selected = dists[k-1:k+3]
    
    # raw = mean of max(d/D, D/d)
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    
    if raw <= 0:
        return 0.0
    return min(1.0, 1.0 / raw)

def _score_polygon(polygon):
    """Compute the overall metric: minimum vertex score."""
    normalized = _normalize(polygon)
    if normalized is None:
        return -1.0  # invalid
    if not _check_convex(normalized):
        return -1.0  # invalid
    
    n = len(normalized)
    scores = [_vertex_score(normalized, i) for i in range(n)]
    return min(scores)

def _soft_min(values, beta=100.0):
    """Smooth approximation of min using log-sum-exp."""
    values = np.array(values, dtype=np.float64)
    # Use the fact that min(x) = -max(-x)
    neg = -values
    max_neg = np.max(neg)
    if max_neg > 500:
        return -max_neg
    exp_neg = np.exp(neg - max_neg)
    log_sum = np.log(np.sum(exp_neg)) + max_neg
    return -log_sum

def _objective(params):
    """Objective function for optimization: maximize min vertex score."""
    n = 20
    # Unpack parameters: 20 x, 20 y
    xs = params[:n]
    ys = params[n:2*n]
    polygon = np.column_stack([xs, ys])
    
    # Normalize
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return -1.0
    normalized = centered / mean_dist
    
    # Check convexity
    if not _check_convex(normalized):
        return -1.0
    
    # Compute vertex scores
    scores = np.array([_vertex_score(normalized, i) for i in range(n)])
    
    # Use soft-min to make objective smooth
    return _soft_min(scores, beta=50.0)

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon that maximizes the minimum vertex score.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin
    
    n = 20
    best_score = -1.0
    best_polygon = None
    
    # Initialize with regular polygon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    regular = np.column_stack([np.cos(angles), np.sin(angles)])
    best_polygon = regular.copy()
    best_score = _score_polygon(regular)
    
    # Multiple restarts with perturbations
    n_restarts = 30
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 60.0:
            break
        
        # Create initial guess: regular polygon with random perturbation
        if restart == 0:
            init = regular.copy()
        else:
            noise_scale = 0.1 * (1 + restart * 0.05)
            init = regular + rng.normal(0, noise_scale, size=(n, 2))
        
        # Flatten to parameter vector
        init_params = init.flatten()
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _objective,
                init_params,
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'maxfun': 5000,
                    'ftol': 1e-10,
                    'gtol': 1e-8
                }
            )
            
            if result.success or result.fun > best_score:
                opt_params = result.x
                opt_polygon = opt_params.reshape(n, 2)
                opt_score = _score_polygon(opt_polygon)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_polygon = opt_polygon.copy()
                    
        except Exception:
            continue
        
        # Also try Nelder-Mead for local polish
        if time.time() > deadline - 30.0:
            break
        
        try:
            result_nm = minimize(
                _objective,
                best_polygon.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 100,
                    'xatol': 1e-10,
                    'fatol': 1e-10
                }
            )
            
            if result_nm.fun > best_score:
                nm_params = result_nm.x
                nm_polygon = nm_params.reshape(n, 2)
                nm_score = _score_polygon(nm_polygon)
                
                if nm_score > best_score:
                    best_score = nm_score
                    best_polygon = nm_polygon.copy()
                    
        except Exception:
            continue
    
    # Final validation and return
    final_normalized = _normalize(best_polygon)
    if final_normalized is not None and _check_convex(final_normalized):
        return final_normalized
    
    # Fallback: return regular polygon
    return regular

# EVOLVE-BLOCK-END
