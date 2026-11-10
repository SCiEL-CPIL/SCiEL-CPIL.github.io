import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.optimize import LinearConstraint

def compute_scores(polygon):
    n = 20
    scores = np.zeros(n)
    for i in range(n):
        v = polygon[i]
        dists = np.linalg.norm(polygon - v, axis=1)
        # Exclude self
        dists[i] = np.inf
        # Sort
        dists = np.sort(dists)
        # d_0 ... d_18
        if dists[0] < 0.05:
            scores[i] = 0.0
            continue
        
        # Find k: first index minimizing d_{k+1} - d_k
        diffs = np.diff(dists)
        k = int(np.argmin(diffs))
        
        D = (dists[k] + dists[k+1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
            
        # Select 4 distances
        if k == 0:
            d_sel = dists[0:4]
        elif k == 17:
            d_sel = dists[15:19]
        else:
            d_sel = dists[k-1:k+3]
            
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
        
    return scores

def _vertex_score(v, pts):
    """Compute the vertex score for vertex v given all points pts."""
    diffs = pts - v
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    # Exclude self (closest point is always self, distance 0)
    # Find index of self
    idx_self = np.argmin(dists)
    mask = np.ones(len(pts), dtype=bool)
    mask[idx_self] = False
    d = np.sort(dists[mask])
    if np.any(d < 0.05):
        return 0.0
    gaps = np.diff(d)
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k - 1:k + 3]
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _min_vertex_score(pts):
    scores = [_vertex_score(pts[i], pts) for i in range(len(pts))]
    return min(scores)

def _is_valid(pts):
    n = len(pts)
    if n != 20:
        return False
    if not np.all(np.isfinite(pts)):
        return False
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-6:
                return False
    # Check convexity
    v_curr = pts
    v_next = np.roll(pts, -1, axis=0)
    v_next2 = np.roll(pts, -2, axis=0)
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    # All must be same sign and nonzero
    # We assume counter-clockwise, so cross > 0
    if np.any(cross <= 1e-6):
        return False
    return True

def _normalize(pts):
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    mean_dist = np.mean(np.linalg.norm(pts, axis=1))
    if mean_dist <= 0.01:
        return None
    return pts / mean_dist

def _param_to_pts(params):
    n = 20
    angles = params[:n]
    radii = params[n:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def _objective(params):
    pts = _param_to_pts(params)
    pts_norm = _normalize(pts)
    if pts_norm is None:
        return 1000.0
    
    # Convexity check
    v_curr = pts_norm
    v_next = np.roll(pts_norm, -1, axis=0)
    v_next2 = np.roll(pts_norm, -2, axis=0)
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    
    # Penalty for non-convexity
    neg_cross = np.minimum(cross, 0)
    penalty = np.sum(neg_cross**2)
    
    score = _min_vertex_score(pts_norm)
    return -score + 10.0 * penalty

def objective(params):
    return _objective(params)

def _soft_min(scores, beta=50.0):
    """
    Approximate min(scores) using log-sum-exp for differentiability.
    min(x) = -max(-x) = -(1/beta) * log(sum(exp(-beta * x)))
    """
    # Handle potential overflow/underflow by shifting
    # We want min, so we maximize -scores
    # Let y = -scores. We want max(y).
    # max(y) ≈ (1/beta) * log(sum(exp(beta * y)))
    # So min(scores) ≈ -(1/beta) * log(sum(exp(-beta * scores)))
    
    # To ensure numerical stability, subtract max(-scores) = -min(scores)
    # This is a bit circular if we don't know min.
    # Standard trick:
    # m = min(scores)
    # min(scores) = m + (1/beta) * log(sum(exp(-beta * (scores - m))))
    # But we want a smooth approximation that works even if we don't know m exactly,
    # or we can just use the raw formula with care.
    
    # Let's use the stable version:
    # Let z = -beta * scores
    # We want - (1/beta) * log(sum(exp(z)))
    # = - (1/beta) * (max(z) + log(sum(exp(z - max(z)))))
    # max(z) = -beta * min(scores)
    
    # For a smooth surrogate, we can just compute:
    # - (1/beta) * log(sum(exp(-beta * scores)))
    # To avoid underflow in exp, we subtract the max exponent term (which corresponds to min score).
    # max_exponent = max(-beta * scores) = -beta * min(scores)
    
    # Since we are using this for optimization, we can just compute it directly if scores are not too large.
    # Scores are in [0, 1]. beta=50.
    # exp(-50 * 1) = exp(-50) ~ 1e-22. Safe.
    # exp(-50 * 0) = 1.
    
    z = -beta * scores
    # To be safe, subtract max(z)
    z_max = np.max(z)
    sum_exp = np.sum(np.exp(z - z_max))
    if sum_exp == 0:
        return -np.inf # Should not happen for finite scores
    log_sum_exp = z_max + np.log(sum_exp)
    return -log_sum_exp / beta

def _smooth_objective(params, beta=50.0):
    pts = _param_to_pts(params)
    pts_norm = _normalize(pts)
    if pts_norm is None:
        return 1000.0
    
    # Convexity penalty
    v_curr = pts_norm
    v_next = np.roll(pts_norm, -1, axis=0)
    v_next2 = np.roll(pts_norm, -2, axis=0)
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    
    neg_cross = np.minimum(cross, 0)
    penalty = np.sum(neg_cross**2)
    
    # Compute scores
    scores = np.array([_vertex_score(pts_norm[i], pts_norm) for i in range(20)])
    
    # Smooth min
    min_score = _soft_min(scores, beta)
    
    return -min_score + 10.0 * penalty

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = 630.0 - 10.0 # Safety margin
    
    n = 20
    # Initial guess: Regular polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radii = np.ones(n)
    x0 = np.concatenate([angles, radii])
    
    best_params = x0
    best_score = -1e9
    
    # 1. Differential Evolution for global search
    # Bounds for angles: [0, 2*pi]
    # Bounds for radii: [0.5, 1.5] (normalized later, but helps keep shape reasonable)
    bounds = [(0, 2*np.pi)] * n + [(0.5, 1.5)] * n
    
    try:
        # Run DE with limited iterations to save time
        # maxiter is number of generations
        # popsize is multiplier for population size
        res_de = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-4, 
            seed=42,
            init='sohs' # Scale hypercube sampling
        )
        
        if res_de.success or res_de.fun < best_score:
            best_params = res_de.x
            best_score = res_de.fun
            
        # Check time
        elapsed = time.time() - start_time
        if elapsed > deadline - 60.0:
             # Not enough time for polish, return best DE result
             pass
        else:
            # 2. Local Polish with Nelder-Mead or L-BFGS-B
            # Use Nelder-Mead on the smooth objective or the original?
            # The original is non-smooth (min). Nelder-Mead is derivative-free and can handle it.
            # But the smooth objective allows L-BFGS-B which might be faster/more accurate.
            # Let's try Nelder-Mead on the original objective first, as it's robust.
            
            res_nm = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if res_nm.fun < best_score:
                best_params = res_nm.x
                best_score = res_nm.fun
                
            # Check time
            elapsed = time.time() - start_time
            if elapsed < deadline - 30.0:
                # 3. Final Polish with L-BFGS-B on Smooth Objective
                # This requires gradient. L-BFGS-B approximates it.
                # We need to pass the smooth objective.
                res_lbfgs = minimize(
                    _smooth_objective, 
                    best_params, 
                    method='L-BFGS-B', 
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                # Verify validity of the L-BFGS result
                pts_lbfgs = _param_to_pts(res_lbfgs.x)
                pts_norm_lbfgs = _normalize(pts_lbfgs)
                if pts_norm_lbfgs is not None and _is_valid(pts_norm_lbfgs):
                    score_lbfgs = -_min_vertex_score(pts_norm_lbfgs)
                    if score_lbfgs > -best_score: # Note: objective returns -score
                        # We should compare the actual min score
                        # best_score is the objective value (negative of score + penalty)
                        # If penalty is 0, best_score = -score
                        # So score = -best_score
                        # If score_lbfgs > score_best, then -score_lbfgs < -score_best
                        # So if res_lbfgs.fun < best_score, it's better
                        if res_lbfgs.fun < best_score:
                            best_params = res_lbfgs.x
                            best_score = res_lbfgs.fun

    except Exception as e:
        # Fallback to regular
        pass

    # Construct final polygon
    angles_opt = best_params[:20]
    radii_opt = best_params[20:]
    
    x = radii_opt * np.cos(angles_opt)
    y = radii_opt * np.sin(angles_opt)
    polygon = np.column_stack((x, y))
    
    # Normalize
    polygon_norm = _normalize(polygon)
    if polygon_norm is None:
         # Fallback to regular
         angles = np.linspace(0, 2*np.pi, n, endpoint=False)
         return np.column_stack((np.cos(angles), np.sin(angles)))
        
    if not _is_valid(polygon_norm):
         # Fallback to regular
         angles = np.linspace(0, 2*np.pi, n, endpoint=False)
         return np.column_stack((np.cos(angles), np.sin(angles)))
         
    return polygon_norm

BET: {mechanism: "Combining global search (Differential Evolution) to escape local minima of the non-convex landscape with a local polish (Nelder-Mead on original, L-BFGS-B on smooth surrogate) should find a better configuration than Nelder-Mead alone starting from a regular polygon.", q: 0.6, delta: [0.0, 0.1]}
