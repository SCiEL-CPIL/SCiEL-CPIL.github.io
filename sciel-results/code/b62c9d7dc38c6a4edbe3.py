# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.special import logsumexp
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    def true_score(points):
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        ratio = d_min / d_max
        return ratio ** 2

    def normalize_points(points):
        """Scale points so max pairwise distance = 1.0, centered at origin."""
        dists = pdist(points, 'euclidean')
        d_max = np.max(dists)
        if d_max < 1e-15:
            return points.copy()
        # Center
        points_centered = points - np.mean(points, axis=0)
        # Scale so max dist = 1
        points_scaled = points_centered / d_max
        return points_scaled

    def soft_min_dist_sq(points, beta):
        """Smooth approximation of min squared distance using log-sum-exp."""
        D2 = pdist(points, 'sqeuclidean')
        # softmin(D2) = exp(-1/beta * logsumexp(-beta * D2))
        log_softmin = -1.0 / beta * logsumexp(-beta * D2)
        return np.exp(log_softmin)

    def objective_fixed_scale(params, beta):
        """
        Objective: minimize -soft_min_dist_sq after normalizing to max_dist=1.
        This eliminates scale invariance.
        """
        points = params.reshape((n, 2))
        points_norm = normalize_points(points)
        # Maximize min distance (equivalent to maximizing min dist^2)
        # Minimize negative of soft min dist^2
        return -soft_min_dist_sq(points_norm, beta)

    def gradient_fixed_scale(params, beta):
        """
        Gradient of -soft_min_dist_sq(normalize(points), beta) w.r.t. params.
        Uses finite differences for robustness given the normalization.
        """
        points = params.reshape((n, 2))
        grad = np.zeros_like(params)
        eps = 1e-7
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            f_plus = objective_fixed_scale(params_plus, beta)
            
            params_minus = params.copy()
            params_minus[i] -= eps
            f_minus = objective_fixed_scale(params_minus, beta)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    def gradient_analytic(params, beta):
        """
        Analytic gradient using chain rule through normalization.
        More efficient than finite differences.
        """
        points = params.reshape((n, 2))
        points_centered = points - np.mean(points, axis=0)
        dists = pdist(points_centered, 'euclidean')
        d_max = np.max(dists)
        
        if d_max < 1e-15:
            return np.zeros_like(params)
        
        points_scaled = points_centered / d_max
        D2 = pdist(points_scaled, 'sqeuclidean')
        
        # Gradient of softmin(D2) w.r.t. D2
        exp_neg_beta_D2 = np.exp(-beta * D2)
        sum_exp_neg = np.sum(exp_neg_beta_D2)
        w = exp_neg_beta_D2 / sum_exp_neg
        
        # Gradient of D2 w.r.t. points_scaled
        idx = np.triu_indices(n, k=1)
        i_idx, j_idx = idx
        
        grad_D2_scaled = np.zeros((n, 2))
        for k in range(len(i_idx)):
            i, j = i_idx[k], j_idx[k]
            diff = points_scaled[i] - points_scaled[j]
            grad_D2_scaled[i] += 2 * w[k] * diff
            grad_D2_scaled[j] -= 2 * w[k] * diff
        
        # Now chain rule: points_scaled = (points - mean(points)) / d_max
        # d(points_scaled)/d(points) = (I - J/n) / d_max where J is all-ones matrix
        # d(d_max)/d(points) = (1/d_max) * sum over pairs of d(d_max)/d(diff_ij)
        
        # For simplicity, use the fact that after normalization,
        # we can compute gradient w.r.t. points_scaled and map back.
        # grad_params = (1/d_max) * (I - J/n) @ grad_D2_scaled
        # But we also need d(d_max)/d(points) term... 
        
        # Simpler: since we normalize, the effective gradient w.r.t. original params
        # is approximately (1/d_max) * (grad_D2_scaled - mean(grad_D2_scaled))
        # (ignoring the d(d_max)/d(points) term for now, which is second-order)
        
        grad_points = (grad_D2_scaled - np.mean(grad_D2_scaled, axis=0)) / d_max
        
        # The objective is -soft_min, so gradient is -grad_points
        return (-grad_points).flatten()

    def refine_local(points, time_limit=5.0):
        """Use Powell method for robust local refinement on non-smooth objective."""
        if time.time() > deadline - 10:
            return points
        
        def obj(p):
            pts = p.reshape((n, 2))
            pts_norm = normalize_points(pts)
            return -soft_min_dist_sq(pts_norm, 1000.0)
        
        start_t = time.time()
        result = minimize(obj, points.flatten(), method='Powell',
                         options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
        
        if time.time() - start_t < time_limit:
            candidate = result.x.reshape((n, 2))
            candidate_norm = normalize_points(candidate)
            if true_score(candidate_norm) > true_score(points):
                return candidate_norm
        return points

    def generate_structured_initializations():
        """Generate diverse structured initializations."""
        inits = []
        
        # 4x4 grid
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        X, Y = np.meshgrid(x, y)
        inits.append(np.column_stack([X.flatten(), Y.flatten()]))
        
        # Hexagonal packing approximation
        # 4 rows of 4 points in hexagonal arrangement
        hex_pts = []
        row_height = np.sqrt(3) / 2
        for row in range(4):
            for col in range(4):
                x_offset = col + (0.5 if row % 2 == 1 else 0)
                hex_pts.append([x_offset, row * row_height])
        inits.append(np.array(hex_pts))
        
        # Concentric circles: 1 center + 5 inner + 10 outer
        angles_inner = np.linspace(0, 2*np.pi, 5, endpoint=False)
        angles_outer = np.linspace(0, 2*np.pi, 10, endpoint=False)
        pts = [[0, 0]]
        for a in angles_inner:
            pts.append([0.4 * np.cos(a), 0.4 * np.sin(a)])
        for a in angles_outer:
            pts.append([0.8 * np.cos(a), 0.8 * np.sin(a)])
        inits.append(np.array(pts))
        
        # Two concentric rings: 8 + 8
        angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        pts = []
        for a in angles1:
            pts.append([0.5 * np.cos(a), 0.5 * np.sin(a)])
        for a in angles2:
            pts.append([1.0 * np.cos(a), 1.0 * np.sin(a)])
        inits.append(np.array(pts))
        
        # Random initializations with different seeds
        rng = np.random.default_rng(42)
        for seed in [42, 123, 456, 789, 1011]:
            rng = np.random.default_rng(seed)
            pts = rng.uniform(-2, 2, size=(n, 2))
            inits.append(pts)
        
        return inits

    initializations = generate_structured_initializations()
    best_points = initializations[0].copy()
    best_score = true_score(best_points)

    # For each initialization, run optimization with increasing beta
    for init_idx, init_points in enumerate(initializations):
        if time.time() > deadline - 60:
            break
        
        current_points = init_points.flatten()
        
        # Smooth optimization with increasing beta
        betas = [50, 100, 200, 500, 1000]
        
        for beta in betas:
            if time.time() > deadline - 30:
                break
            
            def obj_func(p):
                return objective_fixed_scale(p, beta)
            
            # Use analytic gradient when available, else finite differences
            try:
                result = minimize(obj_func, current_points, jac=gradient_analytic, method='L-BFGS-B',
                                 options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if result.success:
                    candidate_points = result.x.reshape((n, 2))
                    candidate_norm = normalize_points(candidate_points)
                    candidate_score = true_score(candidate_norm)
                    
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_points = candidate_norm
                        current_points = candidate_norm.flatten()
                    else:
                        current_points = result.x
            except Exception:
                # Fall back to finite differences if analytic gradient fails
                try:
                    result = minimize(obj_func, current_points, jac=gradient_fixed_scale, method='L-BFGS-B',
                                     options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                    if result.success:
                        candidate_points = result.x.reshape((n, 2))
                        candidate_norm = normalize_points(candidate_points)
                        candidate_score = true_score(candidate_norm)
                        if candidate_score > best_score:
                            best_score = candidate_score
                            best_points = candidate_norm
                            current_points = candidate_norm.flatten()
                except Exception:
                    continue
        
        # Local refinement with Powell
        if time.time() < deadline - 20:
            refined = refine_local(best_points)
            refined_score = true_score(refined)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined

    # Final refinement on the best point found
    if time.time() < deadline - 10:
        final_refined = refine_local(best_points, time_limit=10.0)
        final_score = true_score(final_refined)
        if final_score > best_score:
            best_points = final_refined

    return best_points

# EVOLVE-BLOCK-END
