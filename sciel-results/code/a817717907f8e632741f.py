"""Optimized planar point configuration for maximum product of distances at fixed diameter."""
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
    deadline = time.time() + 145  # Leave 5s safety margin
    n = 8
    
    # Start with regular octagon
    k_vals = np.arange(n)
    base_angle = 2 * np.pi / n
    octagon = np.column_stack([np.cos(base_angle * k_vals), np.sin(base_angle * k_vals)])
    
    best_points = octagon.copy()
    best_score = _compute_score(octagon)
    
    iu = np.triu_indices(n, k=1)
    c = n * (n - 1) // 2
    
    def check_and_update(pts):
        nonlocal best_points, best_score
        min_dist = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                min_dist = min(min_dist, d)
        if min_dist > 1e-6:
            sc = _compute_score(pts)
            if sc > best_score:
                best_score = sc
                best_points = pts.copy()
        return min_dist

    # 1. Smooth Surrogate Objective using soft-max for diameter
    def neg_log_score_smooth(x, beta=100.0):
        """Negative log score with soft-max diameter approximation."""
        pts = x.reshape(-1, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        dsq_upper = dists_sq[iu]
        if np.min(dsq_upper) < 1e-30:
            return 1e10
        
        # Soft-max of squared distances
        m = np.max(dsq_upper)
        # log(sum(exp(dsq/beta))) * beta
        # Numerical stability: subtract max
        log_sum_exp = m + beta * np.log(np.sum(np.exp((dsq_upper - m) / beta)))
        diam_sq_soft = log_sum_exp
        
        if diam_sq_soft < 1e-24:
            return 1e10
            
        log_dsq = np.log(dsq_upper)
        log_sum_dsq = np.sum(log_dsq)
        log_diam_sq = np.log(diam_sq_soft)
        
        # log_prod_ordered = 2*c*log(2) + sum_{i<j} log(dsq_ij) - c*log(diam_sq)
        log_prod_ordered = 2.0 * c * np.log(2.0) + log_sum_dsq - c * log_diam_sq
        return n * np.log(n) - log_prod_ordered

    # 2. Exact Objective with Analytical Gradient
    def neg_log_score_exact(x):
        """Exact negative log score."""
        pts = x.reshape(-1, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        dsq_upper = dists_sq[iu]
        diam_sq = np.max(dsq_upper)
        if diam_sq < 1e-24 or np.min(dsq_upper) < 1e-30:
            return 1e10
        log_sum_dsq = np.sum(np.log(dsq_upper))
        log_diam_sq = np.log(diam_sq)
        log_prod_ordered = 2.0 * c * np.log(2.0) + log_sum_dsq - c * log_diam_sq
        return n * np.log(n) - log_prod_ordered

    def neg_log_score_exact_grad(x):
        """Exact negative log score and its gradient."""
        pts = x.reshape(-1, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        dsq_upper = dists_sq[iu]
        diam_sq = np.max(dsq_upper)
        if diam_sq < 1e-24 or np.min(dsq_upper) < 1e-30:
            return 1e10, np.zeros(2 * n)
        
        # Find which pair(s) achieve the diameter
        # Use a small tolerance to handle numerical noise
        diam_mask = np.isclose(dsq_upper, diam_sq, rtol=1e-9, atol=1e-12)
        
        grad = np.zeros(2 * n)
        
        # Term 1: sum_{i<j} (1/dsq_ij) * d(dsq_ij)/dx_k
        # d(dsq_ij)/dx_k = 2*(pts[i] - pts[j])_k for k in {i*2, i*2+1}
        #                  -2*(pts[i] - pts[j])_k for k in {j*2, j*2+1}
        for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
            diff_ij = pts[i] - pts[j]
            dsq = dsq_upper[idx]
            factor = 1.0 / dsq
            grad[i*2] += 2.0 * factor * diff_ij[0]
            grad[i*2+1] += 2.0 * factor * diff_ij[1]
            grad[j*2] -= 2.0 * factor * diff_ij[0]
            grad[j*2+1] -= 2.0 * factor * diff_ij[1]
        
        # Term 2: - c * (1/diam_sq) * sum_{diam pairs} d(dsq_ij)/dx_k
        # This term only contributes for pairs that are at the diameter
        if np.any(diam_mask):
            for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
                if diam_mask[idx]:
                    diff_ij = pts[i] - pts[j]
                    factor = c / diam_sq
                    grad[i*2] -= 2.0 * factor * diff_ij[0]
                    grad[i*2+1] -= 2.0 * factor * diff_ij[1]
                    grad[j*2] += 2.0 * factor * diff_ij[0]
                    grad[j*2+1] += 2.0 * factor * diff_ij[1]
        
        # neg_log_score = n*log(n) - log_prod_ordered
        # gradient of neg_log_score = - gradient of log_prod_ordered
        grad = -grad
        
        log_sum_dsq = np.sum(np.log(dsq_upper))
        log_diam_sq = np.log(diam_sq)
        log_prod_ordered = 2.0 * c * np.log(2.0) + log_sum_dsq - c * log_diam_sq
        val = n * np.log(n) - log_prod_ordered
        
        return val, grad

    # Optimization Loop
    rng = np.random.default_rng(42)
    n_restarts = 40
    
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Varying noise scales: start small, then larger
        if r < 5:
            noise_scale = 0.05 * (r + 1)
        elif r < 15:
            noise_scale = 0.3 + 0.05 * (r - 5)
        elif r < 25:
            noise_scale = 0.8 + 0.05 * (r - 15)
        else:
            noise_scale = 1.2 + 0.1 * (r - 25)
        
        x0 = octagon.flatten() + rng.normal(0, noise_scale, size=2 * n)
        
        try:
            # Phase 1: Smooth optimization
            res_smooth = minimize(neg_log_score_smooth, x0, method='L-BFGS-B',
                                  options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res_smooth.fun < 1e6:
                pts_smooth = res_smooth.x.reshape(-1, 2)
                check_and_update(pts_smooth)
                
                # Phase 2: Exact gradient-based polish
                # Start from the smooth result
                x_start = res_smooth.x
                res_exact = minimize(neg_log_score_exact_grad, x_start, method='L-BFGS-B',
                                     jac=True,
                                     options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if res_exact.fun < 1e6:
                    pts_exact = res_exact.x.reshape(-1, 2)
                    check_and_update(pts_exact)
                    
                    # Phase 3: Nelder-Mead for final non-smooth adjustments if needed
                    # Only do this if we have time and the gradient method didn't fully converge
                    if time.time() < deadline - 5:
                        res_nm = minimize(neg_log_score_exact, pts_exact.flatten(), method='Nelder-Mead',
                                          options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14})
                        if res_nm.fun < 1e6:
                            pts_nm = res_nm.x.reshape(-1, 2)
                            check_and_update(pts_nm)
                            
        except Exception:
            pass

    # Final verification
    sc = _compute_score(best_points)
    if sc <= 0:
        return [[octagon[i, 0], octagon[i, 1]] for i in range(n)]
    
    # Check coordinates
    if np.max(np.abs(best_points)) >= 1e6:
        return [[octagon[i, 0], octagon[i, 1]] for i in range(n)]
    
    return [[float(best_points[i, 0]), float(best_points[i, 1])] for i in range(n)]
# EVOLVE-BLOCK-END
