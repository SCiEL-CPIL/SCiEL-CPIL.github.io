import numpy as np
import time
from scipy.optimize import minimize

def get_points(deltas):
    """
    Get 20 points on unit circle with cyclic order preserved.
    deltas: 19 values added to angles 1..19 (angle 0 is fixed at 0).
    Angles are wrapped to [0, 2*pi) and sorted to maintain cyclic order.
    """
    n = 20
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    thetas = base_angles.copy()
    if len(deltas) > 0:
        thetas[1:] += deltas
    # Wrap to [0, 2*pi)
    thetas = thetas % (2 * np.pi)
    # Sort to maintain cyclic order
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
        return points
    return centered / mean_dist

def check_convexity(points):
    """Check if polygon is strictly convex."""
    n = len(points)
    if n < 3:
        return False
    
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-10:
                return False
    
    cross_products = []
    for i in range(n):
        p0 = points[i]
        p1 = points[(i+1) % n]
        p2 = points[(i+2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        cross_products.append(cross)
    
    cross_products = np.array(cross_products)
    
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
    D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    if k == 0:
        d_sel = sorted_dists[0:4]
    elif k == 17:
        d_sel = sorted_dists[15:19]
    else:
        d_sel = sorted_dists[k-1:k+3]
        
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw)
    return score

def compute_soft_min_score(deltas, beta):
    """
    Compute negative soft-min of vertex scores.
    Lower is better for the optimizer (since we minimize).
    """
    points = get_points(deltas)
    points = normalize_polygon(points)
    
    n = 20
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = vertex_score(i, points)
    
    # If any score is 0, return a large penalty
    if np.any(scores < 1e-10):
        return 100.0
        
    # Soft-min approximation: -1/beta * log(sum(exp(-beta * s_i)))
    # We want to maximize the min score, so we minimize -soft_min_score
    neg_scores = -beta * scores
    max_neg = np.max(neg_scores)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_scores - max_neg)))
    soft_min = -log_sum / beta
    
    return -soft_min

def compute_hard_score(deltas):
    """Compute the actual min vertex score."""
    points = get_points(deltas)
    points = normalize_polygon(points)
    
    if not check_convexity(points):
        return 0.0
        
    n = 20
    scores = np.array([vertex_score(i, points) for i in range(n)])
    return np.min(scores)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # 600s safety margin
    
    n = 20
    x0 = np.zeros(19)
    
    best_deltas = x0.copy()
    best_score = compute_hard_score(best_deltas)
    best_points = normalize_polygon(get_points(best_deltas))
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Coarse search with moderate beta
    # We use multiple restarts to find a good basin
    num_restarts = 20
    max_iter_coarse = 300
    
    for restart in range(num_restarts):
        if time.time() > deadline - 120:  # Leave time for refinement
            break
            
        if restart == 0:
            x_init = np.zeros(19)
        else:
            # Random perturbations, scaled down for later restarts to focus on best area?
            # Or just random to explore? Let's mix.
            if restart < 5:
                scale = 0.5
            else:
                scale = 0.2
            x_init = rng.normal(0, scale, size=19)
            
        try:
            res = minimize(
                compute_soft_min_score,
                x_init,
                args=(10.0,),  # Beta = 10 for smooth landscape
                method='L-BFGS-B',
                options={'maxiter': max_iter_coarse, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.x is not None:
                current_score = compute_hard_score(res.x)
                if current_score > best_score:
                    best_score = current_score
                    best_deltas = res.x.copy()
                    best_points = normalize_polygon(get_points(best_deltas))
                    
        except Exception:
            pass
            
    # Phase 2: Refinement with higher beta
    # Sharpen the objective to focus on the minimum vertex
    if time.time() < deadline - 60:
        try:
            res_fine = minimize(
                compute_soft_min_score,
                best_deltas,
                args=(50.0,),  # Beta = 50 for sharper focus
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res_fine.x is not None:
                current_score = compute_hard_score(res_fine.x)
                if current_score > best_score:
                    best_score = current_score
                    best_deltas = res_fine.x.copy()
                    best_points = normalize_polygon(get_points(best_deltas))
        except Exception:
            pass
            
    # Phase 3: Coordinate Descent Polish on Hard Objective
    # The hard objective is non-differentiable, so we do a simple coordinate search
    if time.time() < deadline - 10:
        current_deltas = best_deltas.copy()
        current_score = best_score
        
        # Try small perturbations in each dimension
        step_sizes = [0.01, 0.001, 0.0001]
        
        for step in step_sizes:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(19):
                    for direction in [-1, 1]:
                        new_deltas = current_deltas.copy()
                        new_deltas[i] += direction * step
                        new_score = compute_hard_score(new_deltas)
                        
                        if new_score > current_score + 1e-10:
                            current_deltas = new_deltas
                            current_score = new_score
                            improved = True
                            
        if current_score > best_score:
            best_score = current_score
            best_deltas = current_deltas
            best_points = normalize_polygon(get_points(best_deltas))
            
    # Final validation
    final_points = normalize_polygon(get_points(best_deltas))
    if not check_convexity(final_points):
        final_points = normalize_polygon(get_points(np.zeros(19)))
        
    return final_points
