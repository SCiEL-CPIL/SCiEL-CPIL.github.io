"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = 8
    iu = np.triu_indices(n, k=1)
    num_pairs = len(iu[0])
    
    def log_score(pts):
        """Compute log of the score Delta/n^n where Delta is product of distances."""
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        # Product over ordered pairs = (product over unordered pairs)^2
        # So log(Delta) = 2 * sum over i<j log(d_ij)
        log_delta = 2.0 * np.sum(np.log(d_ij))
        # Score = Delta / n^n
        log_score_val = log_delta - n * np.log(n)
        return log_score_val, dists
    
    def diameter(pts):
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        return np.max(dists)
    
    def objective(x, penalty_weight):
        """Objective: minimize negative log score + penalty for diameter > 2"""
        if time.time() > deadline - 5:
            return np.inf
        pts = x.reshape(n, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        
        log_delta = 2.0 * np.sum(np.log(d_ij))
        log_score_val = log_delta - n * np.log(n)
        
        # Smooth penalty for distances exceeding 2
        excess = np.maximum(d_ij - 2.0, 0.0)
        penalty = penalty_weight * np.sum(excess ** 2)
        
        return -log_score_val + penalty
    
    def gradient(x, penalty_weight):
        """Numerical gradient"""
        if time.time() > deadline - 5:
            return np.ones_like(x)
        pts = x.reshape(n, 2)
        eps = 1e-6
        grad = np.zeros_like(x)
        f0 = objective(x, penalty_weight)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (objective(x_plus, penalty_weight) - f0) / eps
        return grad
    
    # Generate diverse initial points
    rng = np.random.RandomState(123)
    
    # Regular octagon
    theta = 2 * np.pi * np.arange(n) / n
    points_oct = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Generate perturbed versions
    initial_points = [points_oct.copy()]
    
    # Add various perturbations
    for scale in [0.05, 0.1, 0.2, 0.3]:
        for _ in range(3):
            perturbed = points_oct + rng.randn(n, 2) * scale
            initial_points.append(perturbed)
    
    # Add some random configurations
    for _ in range(5):
        random_pts = rng.randn(n, 2) * 0.5
        initial_points.append(random_pts)
    
    # Best tracking
    best_log_score = -np.inf
    best_points = points_oct.copy()
    
    # Evaluate initial points
    for pts in initial_points:
        if time.time() > deadline - 20:
            break
        ls, _ = log_score(pts)
        if ls > best_log_score:
            best_log_score = ls
            best_points = pts.copy()
    
    # Optimization with different penalty weights
    penalty_weights = [1.0, 5.0, 10.0, 50.0, 100.0]
    
    for init_pts in initial_points[:8]:  # Limit to 8 initial points for time
        if time.time() > deadline - 15:
            break
        
        for pw in penalty_weights:
            if time.time() > deadline - 10:
                break
            
            x0 = init_pts.flatten()
            
            try:
                result = minimize(
                    lambda x: objective(x, pw),
                    x0,
                    jac=lambda x: gradient(x, pw),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if time.time() > deadline - 5:
                    continue
                
                opt_pts = result.x.reshape(n, 2)
                opt_ls, _ = log_score(opt_pts)
                
                if opt_ls > best_log_score:
                    best_log_score = opt_ls
                    best_points = opt_pts.copy()
                    
            except Exception:
                continue
    
    # Final polish with best penalty weight
    if time.time() < deadline - 20:
        for pw in [10.0, 50.0]:
            if time.time() > deadline - 10:
                break
            x0 = best_points.flatten()
            try:
                result = minimize(
                    lambda x: objective(x, pw),
                    x0,
                    jac=lambda x: gradient(x, pw),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                opt_pts = result.x.reshape(n, 2)
                opt_ls, _ = log_score(opt_pts)
                if opt_ls > best_log_score:
                    best_log_score = opt_ls
                    best_points = opt_pts.copy()
            except Exception:
                pass
    
    # Scale to have diameter exactly 2
    d = diameter(best_points)
    if d > 1e-10:
        best_points = best_points * (2.0 / d)
    
    # Ensure all points are distinct
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(best_points[i] - best_points[j])
            if dist < 1e-8:
                best_points[j] += rng.randn(2) * 1e-4
    
    # Return as list of lists
    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
