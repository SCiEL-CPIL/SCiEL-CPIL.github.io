import numpy as np
import time
import scipy.spatial
import scipy.optimize
from scipy.special import logsumexp

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 600  # Leave 30s margin from 630s
    
    # Precompute triangle indices for efficiency
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    tri_i, tri_j, tri_k = i_idx[mask], j_idx[mask], k_idx[mask]
    
    # Objective function to maximize (exact)
    def objective(points):
        # points shape (13, 2)
        try:
            hull = scipy.spatial.ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 0:
                return 0.0
        except Exception:
            return 0.0
            
        p1 = points[tri_i]
        p2 = points[tri_j]
        p3 = points[tri_k]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        return min_area / A_hull

    # Smooth surrogate using soft-min via log-sum-exp
    # soft_min(areas) = -logsumexp(-log(areas)) but we need to be careful with zeros
    # Use: soft_min = exp(-logsumexp(-areas/beta)) * beta  (approximation)
    # Better: use the fact that min is approximated by -logsumexp(-x/beta)*beta for small beta
    def smooth_objective(points, beta=0.01):
        """Smooth approximation of min_triangle_area / hull_area"""
        try:
            hull = scipy.spatial.ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 0:
                return 0.0
        except Exception:
            return 0.0
            
        p1 = points[tri_i]
        p2 = points[tri_j]
        p3 = points[tri_k]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        # Soft min: as beta -> 0, this approaches min(areas)
        # Use: soft_min = -beta * logsumexp(-areas/beta)
        # For numerical stability, shift by max
        a_max = np.max(areas)
        if a_max < 1e-12:
            return 0.0
        shifted = -areas / beta
        shifted -= np.max(shifted)  # numerical stability
        soft_min = -beta * np.log(np.sum(np.exp(shifted)))
        
        return soft_min / A_hull

    # Negative objectives for minimization
    def neg_objective_exact(x):
        pts = x.reshape(n, 2)
        return -objective(pts)
    
    def neg_objective_smooth(x, beta=0.01):
        pts = x.reshape(n, 2)
        return -smooth_objective(pts, beta)

    # Initial guesses
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init_regular = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Create diverse starting points
    rng = np.random.default_rng(seed=42)
    starting_points = [init_regular.copy()]
    
    # Perturbed regular polygon
    for scale in [0.05, 0.1, 0.2]:
        perturbed = init_regular + rng.normal(0, scale, (n, 2))
        starting_points.append(perturbed)
    
    # Slightly different angular distributions
    for angle_shift in [0.1, 0.2, 0.3]:
        angles2 = np.linspace(0, 2 * np.pi, n, endpoint=False) + angle_shift / n
        alt = np.column_stack([np.cos(angles2), np.sin(angles2)])
        starting_points.append(alt)
    
    # Random convex position (sorted by angle from centroid)
    for seed_val in [1, 2, 3]:
        rng2 = np.random.default_rng(seed=seed_val)
        pts = rng2.uniform(-1, 1, (n, 2))
        # Project to convex position: sort by angle from centroid
        centroid = pts.mean(axis=0)
        angles_pt = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        order = np.argsort(angles_pt)
        pts_sorted = pts[order]
        # Scale to unit circle radius
        distances = np.linalg.norm(pts_sorted - centroid, axis=1)
        max_dist = np.max(distances)
        if max_dist > 0:
            pts_sorted = centroid + (pts_sorted - centroid) / max_dist
        starting_points.append(pts_sorted)
    
    best_score = 0.0
    best_points = init_regular.copy()
    
    # Phase 1: Coarse search with smooth surrogate from multiple starts
    beta_coarse = 0.05
    
    for i, start_pts in enumerate(starting_points):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 60:
            break
            
        x0 = start_pts.flatten()
        
        try:
            # Use smooth objective for coarse search
            res = scipy.optimize.minimize(
                lambda x: neg_objective_smooth(x, beta_coarse),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True}
            )
            
            # Evaluate with exact objective
            pts_res = res.x.reshape(n, 2)
            score_res = objective(pts_res)
            
            if score_res > best_score:
                best_score = score_res
                best_points = pts_res.copy()
                
        except Exception:
            pass
    
    # Phase 2: Refine best point with smaller beta
    if time.time() - start_time < time_limit - 30:
        beta_fine = 0.01
        x_best = best_points.flatten()
        
        try:
            res2 = scipy.optimize.minimize(
                lambda x: neg_objective_smooth(x, beta_fine),
                x_best,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-7, 'adaptive': True}
            )
            
            pts_res2 = res2.x.reshape(n, 2)
            score_res2 = objective(pts_res2)
            
            if score_res2 > best_score:
                best_score = score_res2
                best_points = pts_res2.copy()
                
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish with exact objective
    if time.time() - start_time < time_limit - 10:
        x_final = best_points.flatten()
        
        try:
            res3 = scipy.optimize.minimize(
                neg_objective_exact,
                x_final,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-9, 'adaptive': True}
            )
            
            pts_res3 = res3.x.reshape(n, 2)
            score_res3 = objective(pts_res3)
            
            if score_res3 > best_score:
                best_score = score_res3
                best_points = pts_res3.copy()
                
        except Exception:
            pass
    
    # Phase 4: If time remains, try a few more local refinements from perturbed best
    elapsed = time.time() - start_time
    if elapsed < time_limit - 15:
        rng3 = np.random.default_rng(seed=999)
        for attempt in range(5):
            if time.time() - start_time > time_limit - 10:
                break
                
            perturb_scale = 0.01 * (attempt + 1)
            perturbed_best = best_points + rng3.normal(0, perturb_scale, (n, 2))
            x_pert = perturbed_best.flatten()
            
            try:
                res_p = scipy.optimize.minimize(
                    neg_objective_exact,
                    x_pert,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-8, 'adaptive': True}
                )
                
                pts_p = res_p.x.reshape(n, 2)
                score_p = objective(pts_p)
                
                if score_p > best_score:
                    best_score = score_p
                    best_points = pts_p.copy()
                    
            except Exception:
                pass
    
    # Ensure valid output
    if not np.all(np.isfinite(best_points)):
        best_points = init_regular.copy()
    
    return best_points
