# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(points):
    """Compute Delta/n^n for 8 points, scaled to diameter 2."""
    pts = np.array(points, dtype=float)
    n = len(pts)
    # Compute all pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    # Diameter
    diam = np.max(dists)
    if diam < 1e-12:
        return 0.0
    # Scale to diameter 2
    scale = 2.0 / diam
    scaled_dists = dists * scale
    # Product of all ordered pair distances (i != j)
    # Product over ordered pairs = (product over unordered pairs) ^ 2
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    prod_ordered = 1.0
    for i in range(n):
        for j in range(i+1, n):
            prod_ordered *= scaled_dists[i, j] ** 2
    delta = prod_ordered
    score = delta / (n ** n)
    return score

def construct():
    deadline = time.time() + 140  # Leave 10s safety margin
    n = 8
    
    # Start with regular octagon
    k_vals = np.arange(n)
    base_angle = 2 * np.pi / n
    octagon = np.column_stack([np.cos(base_angle * k_vals), np.sin(base_angle * k_vals)])
    
    best_points = octagon.copy()
    best_score = _compute_score(octagon)
    
    # Smooth objective using soft-max for diameter
    # log(soft_max(x)) = log_sum_exp(x) / beta
    # We want to minimize -log(score)
    # score = prod(d_ij) / diam^n (scaled)
    # log(score) = sum log(d_ij) - n log(diam)
    # Using soft-max diam: diam_soft = exp(log_sum_exp(dists^2) / (2*beta))
    # Let's work with squared distances for stability and speed
    # dists_sq = (pts_i - pts_j)^2
    # diam_sq_soft = exp(log_sum_exp(dists_sq) / beta)
    # log(diam_soft) = 0.5 * log(diam_sq_soft) = 0.5 * (log_sum_exp(dists_sq) / beta)
    # log(prod d_ij) = 0.5 * sum log(dists_sq_ij)
    # Objective to minimize: -log(score) = n*log(n) - log(prod d_ij_scaled)
    # d_ij_scaled = 2 * d_ij / diam
    # log(d_ij_scaled) = log(2) + log(d_ij) - log(diam)
    # sum log(d_ij_scaled) over unordered pairs = C(n,2)*log(2) + 0.5*sum log(dists_sq) - C(n,2)*log(diam)
    # Product over ordered pairs is square of product over unordered pairs.
    # log(Delta) = 2 * [ C(n,2)*log(2) + 0.5*sum log(dists_sq) - C(n,2)*log(diam) ]
    #             = 2*C(n,2)*log(2) + sum log(dists_sq) - 2*C(n,2)*log(diam)
    # log(score) = log(Delta) - n*log(n)
    # neg_log_score = n*log(n) - log(Delta)
    
    beta = 100.0 # Temperature for soft-max
    
    def neg_log_score_smooth(x):
        pts = x.reshape(-1, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        
        # Mask diagonal
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        dists_sq_masked = dists_sq[mask]
        
        # Check for zero distances
        if np.min(dists_sq_masked) < 1e-20:
            return 1e10
            
        # Soft max of dists_sq
        # log_sum_exp(dists_sq) = log(sum(exp(dists_sq)))
        # To avoid overflow, subtract max
        max_dsq = np.max(dists_sq_masked)
        lse = max_dsq + np.log(np.sum(np.exp(dists_sq_masked - max_dsq)))
        log_diam_sq_soft = lse / beta
        log_diam_soft = 0.5 * log_diam_sq_soft
        
        # Sum of log dists_sq
        log_sum_dsq = np.sum(np.log(dists_sq_masked))
        
        c = n * (n - 1) // 2
        log_delta = 2.0 * c * np.log(2.0) + log_sum_dsq - 2.0 * c * log_diam_soft
        neg_log_score = n * np.log(n) - log_delta
        return neg_log_score

    def neg_log_score_exact(x):
        pts = x.reshape(-1, 2)
        sc = _compute_score(pts)
        if sc <= 0:
            return 1e10
        return -np.log(sc)

    rng = np.random.default_rng(42)
    n_restarts = 20 # Increased restarts
    
    for r in range(n_restarts):
        if time.time() > deadline - 40:
            break
        
        # Diverse initializations: Octagon + noise, and some random points
        if r % 2 == 0:
            # Perturbed octagon
            noise_scale = 0.2 * (1 + r * 0.1)
            x0 = octagon.flatten() + rng.normal(0, noise_scale, size=2*n)
        else:
            # Random points in unit circle
            x0 = rng.uniform(-1, 1, size=2*n)
        
        try:
            res = minimize(neg_log_score_smooth, x0, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            
            pts = res.x.reshape(-1, 2)
            
            # Check distinctness
            min_dist = np.inf
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(pts[i] - pts[j])
                    min_dist = min(min_dist, d)
            
            if min_dist > 1e-6:
                sc = _compute_score(pts)
                if sc > best_score:
                    best_score = sc
                    best_points = pts.copy()
                    
        except Exception:
            pass

    # Final polish on the best point found using exact objective
    # Use Nelder-Mead with tight tolerances
    if time.time() < deadline - 10:
        try:
            res = minimize(neg_log_score_exact, best_points.flatten(), method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            pts = res.x.reshape(-1, 2)
            
            min_dist = np.inf
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(pts[i] - pts[j])
                    min_dist = min(min_dist, d)
            
            if min_dist > 1e-8:
                sc = _compute_score(pts)
                if sc > best_score:
                    best_score = sc
                    best_points = pts.copy()
        except Exception:
            pass
            
    # One more polish with a slightly larger step size to escape local minima if needed?
    # Or just return best found.
    
    # Final verification
    sc = _compute_score(best_points)
    if sc <= 0:
        return [[octagon[i, 0], octagon[i, 1]] for i in range(n)]
    
    # Check coordinates
    if np.max(np.abs(best_points)) >= 1e6:
        return [[octagon[i, 0], octagon[i, 1]] for i in range(n)]
    
    return [[float(best_points[i, 0]), float(best_points[i, 1])] for i in range(n)]
# EVOLVE-BLOCK-END
