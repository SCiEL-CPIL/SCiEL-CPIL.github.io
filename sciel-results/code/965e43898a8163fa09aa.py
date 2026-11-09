# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize

def _compute_score(points: np.ndarray) -> float:
    """
    Compute the raw score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 1e-10:
        return 0.0
        
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area <= 0:
        return 0.0
        
    return min_area / A_hull

def _generate_initializations():
    """
    Generate diverse initial configurations for the optimizer.
    """
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    pts = np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])
    inits.append(pts)
    
    # 2. Hexagonal packing approximation
    pts = []
    for x in [0.2, 0.5, 0.8]: pts.append([x, 0.8])
    for x in [0.1, 0.36, 0.63, 0.9]: pts.append([x, 0.53])
    for x in [0.1, 0.36, 0.63, 0.9]: pts.append([x, 0.26])
    for x in [0.3, 0.7]: pts.append([x, 0.0])
    inits.append(np.array(pts))
    
    # 3. Concentric rings
    pts = [[0.5, 0.5]]
    r1 = 0.2
    for i in range(6):
        a = np.pi / 3 * i
        pts.append([0.5 + r1 * np.cos(a), 0.5 + r1 * np.sin(a)])
    r2 = 0.4
    for i in range(6):
        a = np.pi / 3 * i + np.pi / 6
        pts.append([0.5 + r2 * np.cos(a), 0.5 + r2 * np.sin(a)])
    inits.append(np.array(pts))
    
    # 4. Random points
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.random((13, 2)) * 0.8 + 0.1
        inits.append(pts)
        
    return inits

def _soft_min_objective(x, beta):
    """
    Objective function for the soft-min optimization.
    We want to maximize the minimum triangle area relative to the hull area.
    Equivalent to minimizing: log(A_hull) - (1/beta) * log(sum(exp(-beta * A_i)))
    where A_i are the triangle areas.
    The term (1/beta) * log(sum(exp(-beta * A_i))) approximates -min(A_i).
    So we minimize log(A_hull) + (1/beta) * log(sum(exp(-beta * A_i))).
    """
    points = x.reshape(13, 2)
    
    # Calculate Hull Area
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
        if A_hull < 1e-10:
            return 1e10
    except Exception:
        return 1e10
        
    # Calculate Triangle Areas
    idx = np.arange(13)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    if np.min(areas) < 1e-10:
        return 1e10
        
    # Log-sum-exp for stability
    # We want to approximate -min(A).
    # Let f = -beta * A.
    # log(sum(exp(f))) = -beta * min(A) + log(sum(exp(-beta*(A-min(A)))))
    # The second term is bounded.
    # So (1/beta) * log(sum(exp(-beta*A))) approximates -min(A).
    
    # We want to minimize log(A_hull) - min(A).
    # So objective = log(A_hull) + (1/beta) * log(sum(exp(-beta * areas)))
    
    shifted_areas = -beta * areas
    max_shifted = np.max(shifted_areas)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted_areas - max_shifted)))
    
    soft_min_term = (1.0 / beta) * log_sum_exp
    
    return np.log(A_hull) + soft_min_term

def _local_polish(points, start_time, deadline):
    """
    High-precision polish on the true non-smooth objective using gradient ascent.
    """
    current_points = points.copy()
    current_score = _compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    lr = 0.01
    n = 13
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    m = len(triples_arr)
    
    p0_idx = triples_arr[:, 0]
    p1_idx = triples_arr[:, 1]
    p2_idx = triples_arr[:, 2]
    
    steps_since_improve = 0
    
    while time.time() < deadline - 10:
        if steps_since_improve > 500:
            break
            
        # Compute areas
        p0 = current_points[p0_idx]
        p1 = current_points[p1_idx]
        p2 = current_points[p2_idx]
        
        v1 = p1 - p0
        v2 = p2 - p0
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        if min_area < 1e-10:
            current_points += np.random.normal(0, 0.001, current_points.shape)
            current_points = np.clip(current_points, 0.01, 0.99)
            continue
            
        # Bottleneck triangles
        epsilon = 0.05 * min_area
        bottleneck_mask = areas < (min_area + epsilon)
        
        if not np.any(bottleneck_mask):
            bottleneck_mask = np.zeros(m, dtype=bool)
            bottleneck_mask[np.argmin(areas)] = True
            
        grad = np.zeros_like(current_points)
        b_idx = np.where(bottleneck_mask)[0]
        
        for b in b_idx:
            i = p0_idx[b]
            j = p1_idx[b]
            k = p2_idx[b]
            
            pi = current_points[i]
            pj = current_points[j]
            pk = current_points[k]
            
            vi = pj - pi
            vk = pk - pi
            
            D = vi[0] * vk[1] - vi[1] * vk[0]
            
            if D == 0: continue
            
            sign_D = 1.0 if D > 0 else -1.0
            
            # Gradients for i
            grad[i, 0] += sign_D * (2 * pi[0] - pj[0] - pk[0])
            grad[i, 1] += sign_D * (pj[1] - pk[1])
            
            # Gradients for j
            grad[j, 0] += sign_D * (vk[1])
            grad[j, 1] += sign_D * (-vk[0])
            
            # Gradients for k
            grad[k, 0] += sign_D * (-vi[1])
            grad[k, 1] += sign_D * (vi[0])
            
        # Hull gradient (minimize hull area)
        centroid = np.mean(current_points, axis=0)
        hull_grad = current_points - centroid
        hull_weight = 0.05 
        total_grad = grad - hull_weight * hull_grad
        
        norm = np.linalg.norm(total_grad)
        if norm > 1e-10:
            step_vec = lr * total_grad / norm
        else:
            step_vec = np.zeros_like(current_points)
            
        new_points = current_points + step_vec
        new_points = np.clip(new_points, 0.01, 0.99)
        
        new_score = _compute_score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
            steps_since_improve = 0
            lr = min(lr * 1.01, 0.05) # Slightly increase lr if improving
        else:
            steps_since_improve += 1
            lr *= 0.95
            if steps_since_improve % 10 == 0:
                current_points += np.random.normal(0, 0.001, current_points.shape)
                current_points = np.clip(current_points, 0.01, 0.99)
                current_score = _compute_score(current_points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    steps_since_improve = 0

    return best_points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area using a smooth soft-min surrogate and L-BFGS-B.
    """
    start_time = time.time()
    deadline = 630  # seconds, with a safety margin
    
    inits = _generate_initializations()
    
    best_points = None
    best_score = -np.inf
    
    # Precompute constraints for bounds
    bounds = [(0.01, 0.99)] * 26
    
    # We will run L-BFGS-B with increasing beta (sharpness) to converge to the true min.
    # Start with a low beta for smoothness, then increase.
    betas = [50, 100, 200, 500, 1000]
    
    # For each initialization, we run a sequence of optimizations with increasing beta.
    # To save time, we might not run all betas for all inits if time is tight.
    
    for i, init_pts in enumerate(inits):
        if time.time() - start_time > deadline - 60:
            break
            
        current_x = init_pts.flatten()
        
        # Run optimization with increasing beta
        for j, beta in enumerate(betas):
            if time.time() - start_time > deadline - 60:
                break
                
            # Check time for this specific optimization
            # L-BFGS-B can be expensive. We limit maxiter.
            maxiter = 200 if beta < 500 else 100
            
            try:
                res = minimize(
                    _soft_min_objective,
                    current_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                current_x = res.x
            except Exception:
                # If optimization fails, keep current_x
                pass
            
            # Evaluate true score
            current_pts = current_x.reshape(13, 2)
            current_score = _compute_score(current_pts)
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_pts.copy()
        
        # Local polish on the best point found so far from this init
        if time.time() - start_time < deadline - 30:
            polished_pts = _local_polish(current_pts, start_time, deadline)
            polished_score = _compute_score(polished_pts)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_pts.copy()

    # Final global polish if time remains
    if best_points is not None and time.time() - start_time < deadline - 10:
        final_pts = _local_polish(best_points, start_time, deadline)
        final_score = _compute_score(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts.copy()
            
    # Fallback
    if best_points is None:
        best_points = inits[0]
        
    # Final sanity check
    if not np.all(np.isfinite(best_points)):
        best_points = np.array([
            [0.2, 0.8], [0.5, 0.8], [0.8, 0.8],
            [0.1, 0.53], [0.36, 0.53], [0.63, 0.53], [0.9, 0.53],
            [0.1, 0.26], [0.36, 0.26], [0.63, 0.26], [0.9, 0.26],
            [0.3, 0.0], [0.7, 0.0]
        ])
        
    return best_points

# EVOLVE-BLOCK-END
