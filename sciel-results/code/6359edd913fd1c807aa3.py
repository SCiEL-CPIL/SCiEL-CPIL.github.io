# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _score(points):
    """Compute (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = dists.min()
    d_max = dists.max()
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Uses L-BFGS-B on sphere with log-sum-exp soft-min objective,
    plus stochastic perturb-and-relax to escape local minima.
    """
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > (time_limit - safety_margin)
    
    def normalize_sphere(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return points / norms
    
    def pairwise_dists(points):
        """Return all pairwise distances (91 values for 14 points)."""
        return pdist(points)
    
    def soft_min_dists(points, beta=50.0):
        """
        Log-sum-exp approximation to min distance.
        As beta -> inf, this approaches the true min.
        Returns (soft_min_value, gradient_wrt_points).
        """
        dists = pdist(points)
        # Soft min: -beta * log(sum(exp(-beta * dists)))
        # For numerical stability:
        min_d = np.min(dists)
        shifted = -beta * (dists - min_d)
        log_sum = np.log(np.sum(np.exp(shifted)))
        soft_min = min_d - log_sum / beta
        
        # Gradient: d(soft_min)/d(point_k) for each pair
        # soft_min = min_d - (1/beta) * log(sum_j exp(-beta*(d_j - min_d)))
        # = min_d - (1/beta) * log(sum_j exp(-beta*d_j) / exp(-beta*min_d))
        # The gradient of log-sum-exp w.r.t. d_j is exp(-beta*(d_j-min_d)) / sum(exp(-beta*(d_j-min_d)))
        # Then chain rule through d_j = ||p_i - p_k||
        
        exp_shifted = np.exp(shifted)
        weights = exp_shifted / np.sum(exp_shifted)  # softmax weights
        # weights[j] = exp(-beta*(d_j - min_d)) / sum(exp(-beta*(d_j - min_d)))
        
        # d(soft_min)/d(d_j) = -weights[j]  (negative because we're minimizing -log_sum_exp)
        # Actually: soft_min = min_d - (1/beta)*log_sum
        # d(soft_min)/d(min_d) = 1 - (1/beta) * sum(weights[j]) * (-beta) * indicator(j is argmin)
        # This is complex; let's use a simpler approach.
        
        # Alternative: directly compute gradient of -logsumexp(-beta*dists)
        # f = -logsumexp(-beta * dists)
        # df/dd_j = beta * exp(-beta*d_j) / sum(exp(-beta*d_k))
        # For numerical stability:
        max_exp = np.max(-beta * dists)
        exp_vals = np.exp(-beta * dists - max_exp)
        sum_exp = np.sum(exp_vals)
        grad_d = beta * exp_vals / sum_exp  # df/dd_j
        
        # Now chain rule: d_j = ||p_i - p_k||
        # d(d_j)/d(p_i) = (p_i - p_k) / d_j
        # d(d_j)/d(p_k) = (p_k - p_i) / d_j
        
        pts = points
        forces = np.zeros_like(pts)
        
        # Map from pdist index to (i, j) pair
        # pdist ordering: (0,1), (0,2), ..., (0,n-1), (1,2), ...
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                dist = np.sqrt(np.dot(diff, diff))
                if dist < 1e-15:
                    continue
                g = grad_d[idx]
                forces[i] += g * diff / dist
                forces[j] -= g * diff / dist
                idx += 1
        
        return soft_min, forces
    
    def objective_and_grad(flat_points):
        """Objective and gradient for L-BFGS-B (minimize negative soft-min)."""
        pts = flat_points.reshape(n, d)
        pts = normalize_sphere(pts)
        
        # Compute soft min and its gradient
        soft_min_val, forces = soft_min_dists(pts, beta=50.0)
        
        # We want to maximize soft_min, so minimize -soft_min
        obj = -soft_min_val
        grad = -forces.reshape(-1)
        
        return obj, grad
    
    def lbgfs_optimize(init_points, maxiter=200):
        """Run L-BFGS-B optimization starting from init_points."""
        flat_init = init_points.reshape(-1)
        
        result = minimize(
            objective_and_grad,
            flat_init,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-10}
        )
        
        return normalize_sphere(result.x.reshape(n, d))
    
    def fibonacci_sphere(n_points):
        points = []
        for i in range(n_points):
            phi = np.arccos(1 - 2 * (i + 0.5) / n_points)
            theta = np.pi * (1 + 5**0.5) * i
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append([x, y, z])
        return np.array(points)
    
    def random_sphere(n_points, seed):
        rng = np.random.default_rng(seed)
        points = rng.standard_normal((n_points, 3))
        return normalize_sphere(points)
    
    def symmetric_start():
        phi = (1 + 5**0.5) / 2
        icosahedron = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        icosahedron = normalize_sphere(icosahedron)
        
        p13 = normalize_sphere(np.array([[1.0, 1.0, 1.0]]))[0]
        p14 = normalize_sphere(np.array([[-1.0, -1.0, -1.0]]))[0]
        
        return np.vstack([icosahedron, [p13, p14]])
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: Multi-start with L-BFGS-B
    init_configs = [fibonacci_sphere(n), symmetric_start()]
    for seed in range(20):
        init_configs.append(random_sphere(n, seed))
    
    for init_idx, init_pts in enumerate(init_configs):
        if check_deadline():
            break
        
        try:
            optimized = lbgfs_optimize(init_pts, maxiter=150)
            score = _score(optimized)
            if score > best_score:
                best_score = score
                best_points = optimized.copy()
        except Exception:
            pass
    
    # Phase 2: Stochastic perturb-and-relax to escape local minima
    if best_points is not None and not check_deadline():
        rng = np.random.default_rng(123)
        
        for pert_iter in range(50):
            if check_deadline():
                break
            
            # Perturb the best solution
            noise_scale = 0.1 * (1 - pert_iter / 50.0) + 0.01
            perturbed = best_points + rng.standard_normal((n, d)) * noise_scale
            perturbed = normalize_sphere(perturbed)
            
            # Optimize from perturbed state
            try:
                optimized = lbgfs_optimize(perturbed, maxiter=100)
                score = _score(optimized)
                if score > best_score:
                    best_score = score
                    best_points = optimized.copy()
            except Exception:
                pass
    
    # Phase 3: Fine-tuning with higher beta (sharper min approximation)
    if best_points is not None and not check_deadline():
        for beta_val in [100.0, 200.0]:
            if check_deadline():
                break
            
            # Redefine objective with higher beta
            def obj_high_beta(flat_points):
                pts = normalize_sphere(flat_points.reshape(n, d))
                soft_min_val, forces = soft_min_dists(pts, beta=beta_val)
                return -soft_min_val, (-forces).reshape(-1)
            
            try:
                result = minimize(
                    obj_high_beta,
                    best_points.reshape(-1),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 100, 'gtol': 1e-9, 'ftol': 1e-12}
                )
                refined = normalize_sphere(result.x.reshape(n, d))
                score = _score(refined)
                if score > best_score:
                    best_score = score
                    best_points = refined.copy()
            except Exception:
                pass
    
    # Fallback
    if best_points is None:
        best_points = fibonacci_sphere(n)
    
    # Validate
    if not np.all(np.isfinite(best_points)) or best_points.shape != (n, d):
        best_points = fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END
