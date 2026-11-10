# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin

    n_points = 16

    def compute_score(points):
        if points.shape != (n_points, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def normalize_points(points):
        """Scale points such that the maximum pairwise distance is 1."""
        d = pdist(points)
        d_max = np.max(d)
        if d_max > 1e-12:
            return points / d_max
        return points

    def penalty_objective_and_grad(x, beta_min, beta_max, lambda_pen):
        """
        Objective: -soft_min(d) + lambda * (soft_max(d) - 1)^2
        Returns objective value and gradient.
        """
        points = x.reshape(n_points, 2)
        
        # Compute full distance matrix
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=2))
        
        # Mask diagonal
        mask = ~np.eye(n_points, dtype=bool)
        d_flat = d[mask]
        
        if np.max(d_flat) < 1e-12:
            return 0.0, np.zeros_like(x)

        # Soft min
        # s_min = -1/beta_min * log(sum(exp(-beta_min * d)))
        # Use log-sum-exp trick for stability
        min_d = np.min(d_flat)
        exp_terms_min = np.exp(-beta_min * (d_flat - min_d))
        sum_exp_min = np.sum(exp_terms_min)
        s_min = min_d - (1.0 / beta_min) * np.log(sum_exp_min)
        
        # We want to MINIMIZE -s_min
        # So objective part 1 is -s_min
        # Gradient of -s_min w.r.t x_k
        # d(s_min)/dx_k = 1/S * sum_j w_jk * (x_j - x_k)/d_jk
        # So d(-s_min)/dx_k = -1/S * sum_j w_jk * (x_j - x_k)/d_jk = 1/S * sum_j w_jk * (x_k - x_j)/d_jk
        
        weights_min = exp_terms_min / sum_exp_min
        # Reshape weights back to matrix? 
        # It's easier to iterate or use matrix ops.
        # Let's construct the full weight matrix W_min
        W_min = np.zeros((n_points, n_points))
        W_min[mask] = weights_min
        
        # grad_part1[k] = sum_j W_min[k,j] * (x_k - x_j) / d[k,j]
        # Note: W_min is symmetric, d is symmetric.
        # Vectorized:
        # term = W_min * (1/d)  (elementwise, with 0 where d=0)
        # Avoid division by zero
        inv_d = np.zeros_like(d)
        inv_d[mask] = 1.0 / d[mask]
        
        # diff is x_i - x_j
        # We want sum_j W[k,j] * (x_k - x_j) / d[k,j]
        # This is sum_j W[k,j] * diff[k,j] / d[k,j]
        # Let M = W * inv_d
        M = W_min * inv_d
        # grad_part1 = sum_j M[k,j] * diff[k,j]
        # This is matrix multiplication M @ diff? No.
        # diff[k,j] is vector.
        # grad_part1[k] = sum_j M[k,j] * diff[k,j]
        # This is like np.einsum('kij, kj -> ki', M, diff) ?
        # Or simply:
        grad_part1 = np.einsum('kij,kj->ki', M, diff)
        
        # Soft max
        # s_max = 1/beta_max * log(sum(exp(beta_max * d)))
        max_d = np.max(d_flat)
        exp_terms_max = np.exp(beta_max * (d_flat - max_d))
        sum_exp_max = np.sum(exp_terms_max)
        s_max = max_d + (1.0 / beta_max) * np.log(sum_exp_max)
        
        # Penalty term P = lambda * (s_max - 1)^2
        # Grad P = 2 * lambda * (s_max - 1) * grad(s_max)
        # grad(s_max)/dx_k = 1/S_max * sum_j w_jk * (x_k - x_j)/d_jk  (Similar to min but with positive beta)
        # Wait, d(s_max)/dx_k = 1/S * sum w_jk * d(d_jk)/dx_k = 1/S * sum w_jk * (x_k - x_j)/d_jk
        # This is the same form as grad_part1 but with weights_max.
        
        weights_max = exp_terms_max / sum_exp_max
        W_max = np.zeros((n_points, n_points))
        W_max[mask] = weights_max
        M_max = W_max * inv_d
        grad_s_max = np.einsum('kij,kj->ki', M_max, diff)
        
        grad_part2 = 2.0 * lambda_pen * (s_max - 1.0) * grad_s_max
        
        # Total objective
        obj = -s_min + lambda_pen * (s_max - 1.0)**2
        
        # Total gradient
        grad = grad_part1 + grad_part2
        
        return obj, grad.flatten()

    def optimize_candidate(points, time_budget=5.0):
        """
        Optimizes a candidate using L-BFGS-B with penalty method and analytical gradient.
        """
        points = points.copy()
        points = normalize_points(points)
        
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        # Schedule of parameters
        # beta_min, beta_max, lambda_pen
        # Start with low beta (smooth), increase for sharpness.
        # Lambda penalizes deviation from unit diameter.
        params_list = [
            (5, 5, 10.0),
            (10, 10, 10.0),
            (20, 20, 10.0),
            (50, 50, 10.0),
            (100, 100, 10.0),
            (200, 200, 10.0),
            (500, 500, 10.0),
        ]
        
        start_opt = time.time()
        
        for beta_min, beta_max, lambda_pen in params_list:
            if time.time() - start_opt > time_budget:
                break
            
            def obj_grad(x):
                return penalty_objective_and_grad(x, beta_min, beta_max, lambda_pen)
            
            try:
                res = minimize(
                    lambda x: obj_grad(x)[0], 
                    points.flatten(), 
                    jac=lambda x: obj_grad(x)[1],
                    method='L-BFGS-B', 
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                opt_points = res.x.reshape(n_points, 2)
                opt_points = normalize_points(opt_points)
                opt_score = compute_score(opt_points)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
                    
                # Update current points for next stage
                points = best_points.copy()
                
            except Exception as e:
                pass
                
            if time.time() - start_opt > time_budget:
                break
                
        return best_points, best_score

    def polish_candidate(points, time_budget=2.0):
        """
        Final polish using Nelder-Mead on the exact score.
        """
        points = points.copy()
        points = normalize_points(points)
        
        def exact_obj(x):
            pts = x.reshape(n_points, 2)
            # Normalize to keep scale consistent during NM steps?
            # NM moves points. If they drift, score changes.
            # Better to normalize inside obj for NM?
            # NM is derivative-free and robust to scale changes if we normalize.
            pts = normalize_points(pts)
            return -compute_score(pts)
            
        try:
            res = minimize(
                exact_obj, 
                points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-12}
            )
            opt_points = res.x.reshape(n_points, 2)
            opt_points = normalize_points(opt_points)
            opt_score = compute_score(opt_points)
            return opt_points, opt_score
        except Exception:
            return points, compute_score(points)

    best_points = None
    best_score = -np.inf

    # Generate diverse initial candidates
    candidates = []

    # 1. Triangular Lattice
    s = 1.0
    h = s * np.sqrt(3) / 2
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * s + (j % 2) * s / 2
            y = j * h
            lattice_points.append([x, y])
    lattice_points = np.array(lattice_points)
    centroid = np.mean(lattice_points, axis=0)
    dists_to_centroid = np.sum((lattice_points - centroid) ** 2, axis=1)
    idx = np.argsort(dists_to_centroid)[:n_points]
    candidates.append(lattice_points[idx])

    # 2. Square Grid
    grid = np.array([np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4))]).reshape(2, -1).T
    candidates.append(grid)

    # 3. Random Points (Multiple seeds)
    for seed in range(30):
        np.random.seed(seed * 100 + 7)
        random_points = np.random.randn(n_points, 2)
        candidates.append(random_points)

    # 4. Concentric Circles
    # 4 inner, 12 outer
    inner_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False) + np.pi/12
    inner_radius = 0.4
    outer_radius = 1.0
    inner_points = np.column_stack([inner_radius * np.cos(inner_angles), inner_radius * np.sin(inner_angles)])
    outer_points = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
    two_circle = np.vstack([inner_points, outer_points])
    candidates.append(two_circle)
    
    # 5. 6 inner, 10 outer
    inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 10, endpoint=False) + np.pi/10
    inner_radius = 0.45
    outer_radius = 1.0
    inner_points = np.column_stack([inner_radius * np.cos(inner_angles), inner_radius * np.sin(inner_angles)])
    outer_points = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
    two_circle_2 = np.vstack([inner_points, outer_points])
    candidates.append(two_circle_2)
    
    # 6. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    poly_points = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(poly_points)

    # Evaluate and optimize candidates
    processed = []
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 10:
            break
            
        # Optimize each candidate
        opt_cand, opt_score = optimize_candidate(cand, time_budget=5.0)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()
            
        processed.append((opt_score, opt_cand))

    # Sort processed by score
    processed.sort(key=lambda x: x[0], reverse=True)
    
    # Refine the top candidates with polishing
    if best_points is None:
        best_points = candidates[0]
        
    top_candidates = processed[:5]
    for score, cand in top_candidates:
        if time.time() > deadline - 5:
            break
            
        opt_cand, opt_score = polish_candidate(cand, time_budget=2.0)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()

    return best_points

# EVOLVE-BLOCK-END
