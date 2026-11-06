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
        return log_score_val
    
    def diameter(pts):
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        return np.max(dists)
    
    def smooth_max_dist(pts, beta=20.0):
        """Smooth approximation of max pairwise distance using log-sum-exp."""
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        # Use only upper triangle
        sq_dists_iu = sq_dists[iu]
        # log-sum-exp for smooth max
        m = np.max(sq_dists_iu)
        if m <= 0:
            return 0.0
        lse = m / beta + np.log(np.sum(np.exp((sq_dists_iu - m) / beta)))
        return np.sqrt(2.0 * lse / beta)
    
    def objective(x, beta):
        if time.time() > deadline - 3:
            return np.inf
        pts = x.reshape(n, 2)
        log_s = log_score(pts)
        # Smooth diameter constraint: penalize if smooth_max_dist > 2
        smax = smooth_max_dist(pts, beta)
        penalty = 0.0
        if smax > 2.0:
            penalty = (smax - 2.0) ** 2
        return -log_s + penalty
    
    def analytical_gradient(x, beta):
        """Compute gradient of objective analytically."""
        pts = x.reshape(n, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, 1.0)
        
        # Gradient of log_score = 2 * sum_{i<j} log(d_ij)
        # d/dx_k log(d_ij) = (p_i - p_j)_k / d_ij^2
        grad = np.zeros((n, 2))
        for i in range(n):
            for j in range(i+1, n):
                diff = pts[i] - pts[j]
                d_sq = np.dot(diff, diff)
                if d_sq > 1e-12:
                    # Contribution to gradient from pair (i,j)
                    # d log(d_ij)/d p_i = diff / d_sq
                    # d log(d_ij)/d p_j = -diff / d_sq
                    # But we have 2 * log(d_ij) so factor of 2
                    contrib = 2.0 * diff / d_sq
                    grad[i] += contrib
                    grad[j] -= contrib
        
        # Gradient of penalty term
        if beta > 0:
            sq_dists = np.sum(diffs ** 2, axis=2)
            sq_dists_iu = sq_dists[iu]
            m = np.max(sq_dists_iu)
            if m > 1e-12:
                weights = np.exp((sq_dists_iu - m) / beta)
                weights /= np.sum(weights)
                # d smooth_max / d sq_d = weights / (2 * smooth_max)
                smax = smooth_max_dist(pts, beta)
                if smax > 2.0 and smax > 1e-12:
                    penalty_grad = np.zeros((n, 2))
                    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
                        w = weights[idx] / (2.0 * smax)
                        diff = pts[i] - pts[j]
                        penalty_grad[i] += w * diff
                        penalty_grad[j] -= w * diff
                    grad += 2.0 * (smax - 2.0) * penalty_grad
        
        return grad.flatten()
    
    # Regular octagon as baseline
    theta = 2 * np.pi * np.arange(n) / n
    points_oct = np.column_stack([np.cos(theta), np.sin(theta)])
    
    best_score = log_score(points_oct)
    best_points = points_oct.copy()
    
    # Generate diverse starting points
    rng = np.random.RandomState(12345)
    
    # Starting points: regular octagon with various perturbations
    start_points = [points_oct.copy()]
    
    # Small perturbations
    for _ in range(3):
        start_points.append(points_oct + rng.randn(n, 2) * 0.05)
    
    # Medium perturbations
    for _ in range(3):
        start_points.append(points_oct + rng.randn(n, 2) * 0.15)
    
    # Larger perturbations
    for _ in range(2):
        start_points.append(points_oct + rng.randn(n, 2) * 0.3)
    
    # Random points on unit circle with noise
    for _ in range(2):
        angles = np.sort(rng.uniform(0, 2*np.pi, n))
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        start_points.append(pts + rng.randn(n, 2) * 0.1)
    
    # Try different beta values and starting points
    betas = [10.0, 30.0, 100.0]
    
    for sp in start_points:
        if time.time() > deadline - 10:
            break
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
            
            x0 = sp.flatten()
            
            try:
                # Use L-BFGS-B with analytical gradient
                result = minimize(
                    objective,
                    x0,
                    jac=lambda x: analytical_gradient(x, beta),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                
                if time.time() > deadline - 3:
                    continue
                
                opt_pts = result.x.reshape(n, 2)
                opt_score = log_score(opt_pts)
                
                # Check if diameter constraint is satisfied
                d = diameter(opt_pts)
                if d <= 2.001:  # Small tolerance
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_pts.copy()
                        
            except Exception:
                continue
    
    # Final polish with the best point found
    if time.time() < deadline - 15:
        x0 = best_points.flatten()
        for beta in [50.0, 200.0]:
            if time.time() > deadline - 5:
                break
            try:
                result = minimize(
                    objective,
                    x0,
                    jac=lambda x: analytical_gradient(x, beta),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
                )
                
                opt_pts = result.x.reshape(n, 2)
                opt_score = log_score(opt_pts)
                
                d = diameter(opt_pts)
                if d <= 2.001:
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_pts.copy()
            except Exception:
                continue
    
    # Scale to have diameter exactly 2
    d = diameter(best_points)
    if d > 0:
        best_points = best_points * (2.0 / d)
    
    # Final check and return
    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
