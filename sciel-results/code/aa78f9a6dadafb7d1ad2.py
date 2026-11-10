import numpy as np
import time
import sys
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Returns np.ndarray of shape (16, 2).
    """
    start_time = time.time()
    TIME_LIMIT = 630.0
    SAFETY_MARGIN = 15.0
    N = 16

    def time_left():
        return TIME_LIMIT - SAFETY_MARGIN - (time.time() - start_time)

    def get_cost_smooth(params, beta=50.0):
        """
        Smooth approximation of (d_max / d_min)^2 using softmin/softmax.
        Lower is better.
        """
        pts = params.reshape(N, 2)
        # Compute all pairwise squared distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        
        d_min_sq = np.min(dist_sq)
        d_max_sq = np.max(dist_sq)
        
        if d_min_sq <= 1e-15:
            return 1e12
        if d_max_sq <= 1e-15:
            return 1e12
            
        return d_max_sq / d_min_sq

    def get_cost_with_grad(params, beta=50.0):
        """
        Compute cost and gradient using softmin/softmax approximations.
        """
        pts = params.reshape(N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Softmin of dist_sq: -logsumexp(-beta * dist_sq) / beta
        # Softmax of dist_sq: logsumexp(beta * dist_sq) / beta
        # But we need to be careful with inf values
        
        # Mask out diagonal
        mask = ~np.eye(N, dtype=bool)
        dist_sq_masked = dist_sq[mask]
        
        if len(dist_sq_masked) == 0:
            return 1e12, np.zeros_like(params)
        
        d_min_sq = np.min(dist_sq_masked)
        d_max_sq = np.max(dist_sq_masked)
        
        if d_min_sq <= 1e-15 or d_max_sq <= 1e-15:
            return 1e12, np.zeros_like(params)
        
        # Use softmin/softmax for smoothness
        # softmin(x, beta) = -logsumexp(-beta*x)/beta
        # softmax(x, beta) = logsumexp(beta*x)/beta
        
        # For numerical stability
        def softmin(x, beta):
            x_shifted = -beta * x
            x_shifted -= np.max(x_shifted)
            return -np.log(np.sum(np.exp(x_shifted))) / beta
        
        def softmax(x, beta):
            x_shifted = beta * x
            x_shifted -= np.max(x_shifted)
            return np.log(np.sum(np.exp(x_shifted))) / beta
        
        d_min_soft = softmin(dist_sq_masked, beta)
        d_max_soft = softmax(dist_sq_masked, beta)
        
        cost = d_max_soft / d_min_soft
        
        # Gradient via finite differences would be too slow; use analytic gradient
        # d(softmin)/d(x_i) = exp(-beta*x_i) / sum(exp(-beta*x_j))  (for softmin)
        # d(softmax)/d(x_i) = exp(beta*x_i) / sum(exp(beta*x_j))  (for softmax)
        
        # For softmin: gradient wrt x_i = exp(-beta*x_i) / sum(exp(-beta*x_j))
        # For softmax: gradient wrt x_i = exp(beta*x_i) / sum(exp(beta*x_j))
        
        neg_beta_x = -beta * dist_sq_masked
        neg_beta_x -= np.max(neg_beta_x)
        exp_neg = np.exp(neg_beta_x)
        sum_exp_neg = np.sum(exp_neg)
        grad_softmin = exp_neg / sum_exp_neg  # d(softmin)/d(x_i)
        
        beta_x = beta * dist_sq_masked
        beta_x -= np.max(beta_x)
        exp_pos = np.exp(beta_x)
        sum_exp_pos = np.sum(exp_pos)
        grad_softmax = exp_pos / sum_exp_pos  # d(softmax)/d(x_i)
        
        # d(cost)/d(x_i) = grad_softmax_i / d_min_soft - (d_max_soft / d_min_soft^2) * grad_softmin_i
        grad_cost = grad_softmax / d_min_soft - (d_max_soft / (d_min_soft ** 2)) * grad_softmin
        
        # Now map gradient from dist_sq space to params space
        # dist_sq[i,j] = sum_k (pts[i,k] - pts[j,k])^2
        # d(dist_sq[i,j])/d(pts[i,k]) = 2*(pts[i,k] - pts[j,k])
        # d(dist_sq[i,j])/d(pts[j,k]) = -2*(pts[i,k] - pts[j,k])
        
        grad_params = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                idx = i * N + j
                g = grad_cost[idx]
                diff_ij = pts[i] - pts[j]
                grad_params[i] += 2 * g * diff_ij
                grad_params[j] -= 2 * g * diff_ij
        
        return cost, grad_params.flatten()

    def get_cost_and_grad(params):
        cost, grad = get_cost_with_grad(params, beta=100.0)
        return cost, grad

    # Initial configurations
    def regular_polygon(n, radius=1.0):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

    def hex_grid(n):
        """Generate points on a hexagonal grid."""
        points = []
        # Try different grid sizes
        for side in range(1, 5):
            for dx in range(-side, side + 1):
                for dy in range(-side, side + 1):
                    # Hex grid spacing
                    x = dx + 0.5 * (dy % 2)
                    y = dy * np.sqrt(3) / 2
                    points.append([x, y])
            if len(points) >= n:
                break
        points = np.array(points[:n])
        # Normalize
        center = np.mean(points, axis=0)
        points = points - center
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if max_dist > 0:
            points = points / max_dist
        return points

    def random_points(n, seed=0):
        rng = np.random.default_rng(seed)
        return rng.uniform(-1, 1, size=(n, 2))

    # Generate diverse initial points
    initial_points = []
    initial_points.append(regular_polygon(N).flatten())
    initial_points.append(hex_grid(N).flatten())
    for seed in range(10):
        initial_points.append(random_points(N, seed=seed).flatten())

    best_result = None
    best_cost = np.inf

    # First, try DE for global search
    if time_left() > 60:
        try:
            bounds = [(-1.0, 1.0)] * (N * 2)
            de_res = differential_evolution(
                get_cost_smooth,
                bounds,
                maxiter=200,
                popsize=30,
                tol=1e-8,
                mutation=(0.5, 1.5),
                recombination=0.7,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            if de_res.fun < best_cost:
                best_cost = de_res.fun
                best_result = de_res.x.copy()
        except Exception as e:
            print(f"DE failed: {e}", file=sys.stderr)

    # Local refinement from multiple starts
    starts = list(initial_points)
    if best_result is not None:
        starts.append(best_result)

    for start in starts:
        if time_left() < 10:
            break
        try:
            # Try L-BFGS-B with gradient
            res = minimize(
                get_cost_and_grad,
                start,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
            )
            if res.fun < best_cost:
                best_cost = res.fun
                best_result = res.x.copy()
        except Exception as e:
            pass

        # Also try Nelder-Mead for robustness
        if time_left() < 10:
            break
        try:
            nm_res = minimize(
                get_cost_smooth,
                start,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if nm_res.fun < best_cost:
                best_cost = nm_res.fun
                best_result = nm_res.x.copy()
        except Exception as e:
            pass

    # Additional local restarts from best result
    if best_result is not None and time_left() > 30:
        rng = np.random.default_rng(123)
        for i in range(20):
            if time_left() < 15:
                break
            spread = np.std(best_result)
            if spread < 1e-6:
                spread = 0.1
            noise = rng.normal(0, spread * 0.05, best_result.shape)
            perturbed = best_result + noise
            try:
                res = minimize(
                    get_cost_and_grad,
                    perturbed,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                if res.fun < best_cost:
                    best_cost = res.fun
                    best_result = res.x.copy()
            except Exception:
                pass

    # Fallback
    if best_result is None or not np.all(np.isfinite(best_result)):
        best_result = regular_polygon(N).flatten()

    return best_result.reshape(16, 2)
