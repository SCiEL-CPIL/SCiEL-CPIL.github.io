import numpy as np
import time
import scipy.spatial
import scipy.optimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n=13 points on or inside a convex region to maximize
    the area of the smallest triangle formed by these points, normalized by the convex hull area.
    
    Returns:
        points: np.ndarray of shape (13, 2)
    """
    n = 13
    start_time = time.time()
    time_limit = 590  # Leave 40s margin from 630s

    # Precompute triangle indices
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    tri_i, tri_j, tri_k = i_idx[mask], j_idx[mask], k_idx[mask]
    num_tris = len(tri_i)

    def compute_ratio(points):
        points = np.asarray(points, dtype=float)
        if points.shape != (n, 2):
            return 0.0
        if np.any(~np.isfinite(points)):
            return 0.0
        try:
            hull = scipy.spatial.ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-12:
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

    def soft_min_ratio(x, beta=50.0):
        """
        Smooth surrogate: -softmin(areas) / A_hull
        softmin(areas) = -log(sum(exp(-areas/beta))) * beta
        This is smooth and differentiable.
        """
        pts = x.reshape(n, 2)
        
        # Compute areas
        p1 = pts[tri_i]
        p2 = pts[tri_j]
        p3 = pts[tri_k]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        # Soft min via log-sum-exp
        # softmin = -beta * log(sum(exp(-areas/beta)))
        neg_areas_scaled = -areas / beta
        max_neg = np.max(neg_areas_scaled)
        log_sum = max_neg + np.log(np.sum(np.exp(neg_areas_scaled - max_neg)))
        soft_min_area = -beta * log_sum
        
        # Compute hull area
        try:
            hull = scipy.spatial.ConvexHull(pts)
            A_hull = hull.volume
            if A_hull <= 1e-12:
                return 1e10  # Large penalty
        except Exception:
            return 1e10
        
        return -soft_min_area / A_hull

    def soft_min_ratio_grad(x, beta=50.0):
        """
        Gradient of soft_min_ratio.
        """
        pts = x.reshape(n, 2)
        
        # Compute areas
        p1 = pts[tri_i]
        p2 = pts[tri_j]
        p3 = pts[tri_k]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        # Soft min
        neg_areas_scaled = -areas / beta
        max_neg = np.max(neg_areas_scaled)
        exp_vals = np.exp(neg_areas_scaled - max_neg)
        sum_exp = np.sum(exp_vals)
        probs = exp_vals / sum_exp  # weights for soft min
        
        # d(soft_min)/d(area_i) = -probs[i]
        # d(area_i)/d(x) for each triangle
        
        # Compute hull area
        try:
            hull = scipy.spatial.ConvexHull(pts)
            A_hull = hull.volume
            if A_hull <= 1e-12:
                return np.ones_like(x)
        except Exception:
            return np.ones_like(x)
        
        # Gradient of -soft_min_area / A_hull = -(d(soft_min)/dx * A_hull - soft_min * d(A_hull)/dx) / A_hull^2
        # Simplify: we'll compute gradient of soft_min_area w.r.t. x, then scale
        # d(soft_min_area)/d(area_i) = -probs[i]
        # area_i = 0.5 * |cross_i|
        # For simplicity, assume all crosses are positive (typical in good configurations)
        # d(area_i)/d(x_a) where x_a is coordinate of point a
        
        grad_soft_min = np.zeros_like(pts)
        
        for t in range(num_tris):
            w = -probs[t]  # d(soft_min)/d(area_t)
            if abs(w) < 1e-15:
                continue
            
            i, j, k = tri_i[t], tri_j[t], tri_k[t]
            
            # area = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
            # Assuming positive cross: area = 0.5 * cross
            # d(area)/d(x_i) = 0.5 * (y_j - y_k)
            # d(area)/d(y_i) = 0.5 * (x_k - x_j)
            # d(area)/d(x_j) = 0.5 * (y_k - y_i)
            # d(area)/d(y_j) = 0.5 * (x_i - x_k)
            # d(area)/d(x_k) = 0.5 * (y_i - y_j)
            # d(area)/d(y_k) = 0.5 * (x_j - x_i)
            
            yi, yj, yk = pts[i, 1], pts[j, 1], pts[k, 1]
            xi, xj, xk = pts[i, 0], pts[j, 0], pts[k, 0]
            
            # d(area)/d(x_i)
            grad_soft_min[i, 0] += w * 0.5 * (yj - yk)
            # d(area)/d(y_i)
            grad_soft_min[i, 1] += w * 0.5 * (xk - xj)
            # d(area)/d(x_j)
            grad_soft_min[j, 0] += w * 0.5 * (yk - yi)
            # d(area)/d(y_j)
            grad_soft_min[j, 1] += w * 0.5 * (xi - xk)
            # d(area)/d(x_k)
            grad_soft_min[k, 0] += w * 0.5 * (yi - yj)
            # d(area)/d(y_k)
            grad_soft_min[k, 1] += w * 0.5 * (xj - xi)
        
        # Now we have d(soft_min_area)/d(x) = grad_soft_min
        # We want gradient of -soft_min_area / A_hull
        # = -1/A_hull * grad_soft_min + soft_min_area/A_hull^2 * d(A_hull)/d(x)
        # The second term involves gradient of convex hull area, which is complex.
        # For now, approximate by ignoring the d(A_hull)/d(x) term (common heuristic)
        # This gives us a descent direction for the main term.
        
        soft_min_area = -beta * (max_neg + np.log(sum_exp))
        
        # Approximate gradient: -grad_soft_min / A_hull
        grad = -grad_soft_min / A_hull
        
        return grad.flatten()

    # 1. Generate structured initial guesses
    guesses = []

    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    guesses.append(np.column_stack([np.cos(angles), np.sin(angles)]))

    # 12-gon + Center
    angles12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    poly12 = np.column_stack([np.cos(angles12), np.sin(angles12)])
    guesses.append(np.vstack([poly12, [[0.0, 0.0]]]))

    # 11-gon + Center + 1 point
    angles11 = np.linspace(0, 2 * np.pi, 11, endpoint=False)
    poly11 = np.column_stack([np.cos(angles11), np.sin(angles11)])
    guesses.append(np.vstack([poly11, [[0.0, 0.0]], [[1.0, 0.0]]]))

    # Two concentric hexagons (rotated) + Center
    # 6 + 6 + 1 = 13
    angles_h1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    poly_h1 = np.column_stack([np.cos(angles_h1), np.sin(angles_h1)])
    angles_h2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    poly_h2 = np.column_stack([0.5 * np.cos(angles_h2), 0.5 * np.sin(angles_h2)])
    guesses.append(np.vstack([poly_h1, poly_h2, [[0.0, 0.0]]]))

    # Random initializations (a few)
    rng = np.random.default_rng(seed=42)
    for _ in range(3):
        guesses.append(rng.normal(0, 1, (n, 2)))

    best_score = -np.inf
    best_points = None

    # Phase 1: L-BFGS-B with soft-min surrogate on each guess
    for g in guesses:
        if time.time() - start_time > time_limit - 60:
            break
            
        try:
            x0 = g.flatten()
            # Annotate with gradient
            def objective_with_grad(x):
                val = soft_min_ratio(x, beta=30.0)
                grad = soft_min_ratio_grad(x, beta=30.0)
                return val, grad
            
            res = scipy.optimize.minimize(
                objective_with_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            pts = res.x.reshape(n, 2)
            score = compute_ratio(pts)
            if score > best_score:
                best_score = score
                best_points = pts
        except Exception:
            continue

    if best_points is None:
        # Fallback to regular 13-gon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = compute_ratio(best_points)

    # Phase 2: Refine with decreasing beta (annealing) using L-BFGS-B
    if time.time() - start_time < time_limit - 30:
        betas = [30.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3]
        for beta in betas:
            if time.time() - start_time > time_limit - 10:
                break
            try:
                x0 = best_points.flatten()
                def objective_beta(x):
                    val = soft_min_ratio(x, beta=beta)
                    grad = soft_min_ratio_grad(x, beta=beta)
                    return val, grad
                
                res = scipy.optimize.minimize(
                    objective_beta,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                pts = res.x.reshape(n, 2)
                score = compute_ratio(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts
            except Exception:
                continue

    # Phase 3: Final polish with Nelder-Mead on the exact objective
    if time.time() - start_time < time_limit - 5:
        try:
            x0 = best_points.flatten()
            res = scipy.optimize.minimize(
                lambda x: -compute_ratio(x.reshape(n, 2)),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            pts = res.x.reshape(n, 2)
            score = compute_ratio(pts)
            if score > best_score:
                best_score = score
                best_points = pts
        except Exception:
            pass

    # Phase 4: Random restarts with L-BFGS-B if time allows
    if time.time() - start_time < time_limit - 5:
        rng2 = np.random.default_rng(seed=99)
        for _ in range(3):
            if time.time() - start_time > time_limit - 5:
                break
            perturbed = best_points + rng2.normal(0, 0.03, (n, 2))
            x0 = perturbed.flatten()
            try:
                def objective_restart(x):
                    val = soft_min_ratio(x, beta=10.0)
                    grad = soft_min_ratio_grad(x, beta=10.0)
                    return val, grad
                
                res = scipy.optimize.minimize(
                    objective_restart,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                pts = res.x.reshape(n, 2)
                score = compute_ratio(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts
            except Exception:
                continue

    return best_points
