import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def get_points(deltas):
    """Get 20 points on unit circle with cyclic order preserved.
    deltas: 19 values added to angles 1..19 (angle 0 is fixed at 0).
    Angles are wrapped to [0, 2*pi) and sorted to maintain cyclic order."""
    n = 20
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    thetas = base_angles.copy()
    if len(deltas) > 0:
        thetas[1:] += deltas
    # Wrap to [0, 2*pi)
    thetas = thetas % (2 * np.pi)
    # Sort to maintain cyclic order (this preserves the polygon structure)
    thetas = np.sort(thetas)
    x = np.cos(thetas)
    y = np.sin(thetas)
    return np.column_stack((x, y))

def normalize_polygon(points):
    """Normalize: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return points  # Return as-is if degenerate
    return centered / mean_dist

def check_convexity(points):
    """Check if polygon is strictly convex: no repeated points, all turns same sign, total turn = ±2π."""
    n = len(points)
    if n < 3:
        return False
    
    # Check no repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-10:
                return False
    
    # Compute cross products to check turn direction
    cross_products = []
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        cross_products.append(cross)
    
    cross_products = np.array(cross_products)
    
    # All cross products must be nonzero and same sign
    if np.any(np.abs(cross_products) < 1e-10):
        return False
    
    signs = np.sign(cross_products)
    if len(np.unique(signs)) != 1:
        return False
    
    return True

def vertex_score(v_idx, points):
    """Compute score for vertex v_idx given normalized points."""
    dists = np.linalg.norm(points - points[v_idx], axis=1)
    dists[v_idx] = np.inf
    sorted_dists = np.sort(dists)
    if sorted_dists[0] < 0.05:
        return 0.0
    gaps = sorted_dists[1:] - sorted_dists[:-1]
    k = np.argmin(gaps)
    D = (sorted_dists[k] + sorted_dists[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        d_sel = sorted_dists[0:4]
    elif k == 17:
        d_sel = sorted_dists[15:19]
    else:
        d_sel = sorted_dists[k - 1:k + 3]
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw)
    return score

def compute_soft_min_score(deltas, beta=20.0):
    """Compute negative soft-min of vertex scores for smooth optimization.
    Uses log-sum-exp trick for numerical stability."""
    points = get_points(deltas)
    points = normalize_polygon(points)
    
    n = 20
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = vertex_score(i, points)
    
    # If any score is 0, the min is 0
    if np.any(scores < 1e-10):
        return 0.0
    
    # Soft-min: -log-sum-exp of negative scores, scaled by beta
    # soft_min = -(1/beta) * log(sum(exp(-beta * s_i)))
    neg_scores = -beta * scores
    max_neg = np.max(neg_scores)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_scores - max_neg)))
    soft_min = -log_sum / beta
    
    return soft_min

def compute_hard_score(deltas):
    """Compute the actual min vertex score (for final evaluation)."""
    points = get_points(deltas)
    points = normalize_polygon(points)
    
    if not check_convexity(points):
        return 0.0
    
    n = 20
    scores = np.array([vertex_score(i, points) for i in range(n)])
    return np.min(scores)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # safety margin
    
    n = 20
    x0 = np.zeros(19)
    
    # Initialize with regular polygon
    best_deltas = x0.copy()
    best_points = get_points(x0)
    best_points = normalize_polygon(best_points)
    best_score = compute_hard_score(best_deltas)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Differential Evolution for global search
    # This explores the 19-dimensional space more robustly than random restarts
    if time.time() < deadline - 120:
        try:
            def de_objective(deltas):
                return -compute_soft_min_score(deltas, beta=10.0)
            
            bounds = [(-0.5, 0.5)] * 19
            de_result = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-5,
                polish=False
            )
            
            if de_result.x is not None:
                current_score = compute_hard_score(de_result.x)
                if current_score > best_score:
                    best_score = current_score
                    best_deltas = de_result.x.copy()
                    best_points = normalize_polygon(get_points(de_result.x))
                    
        except Exception:
            pass
    
    # Phase 2: Coarse search with multiple restarts using L-BFGS-B
    num_restarts = 30
    max_iter_coarse = 500
    
    for restart in range(num_restarts):
        if time.time() > deadline - 90:
            break
        
        if restart == 0:
            x_init = np.zeros(19)
        elif restart == 1:
            # Start from DE result if it was good
            x_init = best_deltas.copy()
        else:
            # Diverse initializations
            scale = 0.3 / (1 + restart * 0.1)
            x_init = rng.normal(0, scale, size=19)
        
        try:
            res = minimize(
                compute_soft_min_score,
                x_init,
                args=(20.0,),
                method='L-BFGS-B',
                options={'maxiter': max_iter_coarse, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            
            if res.x is not None:
                current_score = compute_hard_score(res.x)
                if current_score > best_score:
                    best_score = current_score
                    best_deltas = res.x.copy()
                    best_points = normalize_polygon(get_points(res.x))
                    
        except Exception:
            pass
    
    # Phase 3: Fine-tuning from best point with higher precision
    if time.time() < deadline - 60:
        try:
            res_fine = minimize(
                compute_soft_min_score,
                best_deltas,
                args=(50.0,),  # Higher beta for sharper soft-min
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res_fine.x is not None:
                current_score = compute_hard_score(res_fine.x)
                if current_score > best_score:
                    best_score = current_score
                    best_deltas = res_fine.x.copy()
                    best_points = normalize_polygon(get_points(res_fine.x))
        except Exception:
            pass
    
    # Phase 4: Local polish with Nelder-Mead on the 19-dim parameter space
    if time.time() < deadline - 30:
        try:
            res_nm = minimize(
                lambda x: -compute_hard_score(x),
                best_deltas,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res_nm.x is not None:
                current_score = compute_hard_score(res_nm.x)
                if current_score > best_score:
                    best_score = current_score
                    best_deltas = res_nm.x.copy()
                    best_points = normalize_polygon(get_points(res_nm.x))
        except Exception:
            pass
    
    # Final: ensure we return valid normalized points
    final_points = normalize_polygon(get_points(best_deltas))
    
    # Final validation
    if not check_convexity(final_points):
        # Fallback to regular polygon
        final_points = normalize_polygon(get_points(np.zeros(19)))
    
    return final_points
