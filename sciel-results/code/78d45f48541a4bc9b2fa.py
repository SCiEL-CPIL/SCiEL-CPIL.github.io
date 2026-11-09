"""Optimized planar point configuration for maximum product of distances at fixed diameter."""
# EVOLVE-BLOCK-START
import numpy as np
import time

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
    # Product of distances over ordered pairs = product over i<j of d_ij^2
    # Since ordered pairs (i,j) for i!=j, that's 2 * (number of unordered pairs)
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

def _neg_log_score(x):
    """Negative log of score for minimization. x is 16-dim flat array."""
    pts = x.reshape(-1, 2)
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    diam = np.max(dists)
    if diam < 1e-12:
        return 1e10
    scale = 2.0 / diam
    # Log of product of ordered pair distances
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    log_prod = 0.0
    for i in range(n):
        for j in range(i+1, n):
            log_prod += 2.0 * np.log(scaled_dists[i, j]) if scaled_dists[i,j] > 1e-15 else -1e10
    # score = exp(log_prod) / n^n
    # log(score) = log_prod - n*log(n)
    # neg_log_score = -log(score) = n*log(n) - log_prod
    if scaled_dists.min() < 1e-15:
        return 1e10
    log_prod = 0.0
    for i in range(n):
        for j in range(i+1, n):
            log_prod += 2.0 * np.log(scaled_dists[i, j])
    return n * np.log(n) - log_prod

def construct():
    deadline = time.time() + 140  # Leave 10s safety margin
    n = 8
    
    # Start with regular octagon
    k_vals = np.arange(n)
    base_angle = 2 * np.pi / n
    octagon = np.column_stack([np.cos(base_angle * k_vals), np.sin(base_angle * k_vals)])
    
    best_points = octagon.copy()
    best_score = _compute_score(octagon)
    
    # Try to improve using optimization
    # Use log-product objective (smooth)
    def neg_log_score_vec(x):
        pts = x.reshape(-1, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        diam_sq = np.max(dists_sq)
        if diam_sq < 1e-24:
            return 1e10
        # Scale: distances become sqrt(dists_sq) * 2/sqrt(diam_sq) = 2*sqrt(dists_sq/diam_sq)
        # Log product of ordered pairs = sum over i<j of 2*log(2*sqrt(dists_sq[i,j]/diam_sq))
        # = sum over i<j of 2*(log(2) + 0.5*log(dists_sq[i,j]/diam_sq))
        # = 2*sum_{i<j}(log(2) + 0.5*log(dists_sq[i,j])) - 2*sum_{i<j}*0.5*log(diam_sq)
        # = 2*C(n,2)*log(2) + sum_{i<j} log(dists_sq[i,j]) - C(n,2)*log(diam_sq)
        c = n * (n - 1) // 2
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        # Use upper triangle for efficiency
        iu = np.triu_indices(n, k=1)
        dsq = dists_sq[iu]
        if np.min(dsq) < 1e-30:
            return 1e10
        log_sum = np.sum(np.log(dsq))
        log_diam = 0.5 * np.log(diam_sq)
        log_prod_ordered = 2.0 * c * np.log(2.0) + log_sum - c * np.log(diam_sq)
        return n * np.log(n) - log_prod_ordered
    
    # Multiple random restarts with perturbation
    rng = np.random.default_rng(42)
    n_restarts = 12
    
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Start from octagon with small random perturbation
        noise_scale = 0.1 * (r + 1)
        x0 = octagon.flatten() + rng.normal(0, noise_scale, size=2*n)
        
        try:
            from scipy.optimize import minimize
            res = minimize(neg_log_score_vec, x0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success or res.fun < 1e6:
                pts = res.x.reshape(-1, 2)
                # Check distinctness
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
    
    # Nelder-Mead polish on exact score (non-smooth due to max)
    def neg_score_exact(x):
        pts = x.reshape(-1, 2)
        sc = _compute_score(pts)
        if sc <= 0:
            return 1e10
        return -np.log(sc)
    
    if time.time() < deadline - 20:
        try:
            from scipy.optimize import minimize
            res = minimize(neg_score_exact, best_points.flatten(), method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
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
    
    # Final verification
    sc = _compute_score(best_points)
    if sc <= 0:
        return [[octagon[i, 0], octagon[i, 1]] for i in range(n)]
    
    # Check coordinates
    if np.max(np.abs(best_points)) >= 1e6:
        return [[octagon[i, 0], octagon[i, 1]] for i in range(n)]
    
    return [[float(best_points[i, 0]), float(best_points[i, 1])] for i in range(n)]
# EVOLVE-BLOCK-END
