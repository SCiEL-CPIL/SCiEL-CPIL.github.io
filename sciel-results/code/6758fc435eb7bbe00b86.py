import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n = 8
    # Regular octagon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = np.ones(n)
    x0 = np.column_stack((radii * np.cos(angles), radii * np.sin(angles))).flatten()
    
    def pairwise_distances(x):
        pts = x.reshape(n, 2)
        # Compute all pairwise distances
        # Using broadcasting
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        return dists

    def log_score(x, k=50):
        dists = pairwise_distances(x)
        # Get upper triangle indices
        iu = np.triu_indices(n, 1)
        d = dists[iu]
        # Avoid log(0)
        d = np.maximum(d, 1e-10)
        
        log_prod = np.sum(np.log(d))
        
        # Soft max for diameter
        # Sum of d^k
        # d^k can overflow if d is large and k is large, but coordinates are bounded.
        # We can scale d first? No, the objective is scale invariant in a way, 
        # but the smooth term depends on the scale relative to k.
        # Actually, the score is scale invariant. 
        # Let's check: if we scale points by c, d -> cd.
        # log_prod -> log_prod + 28 log c.
        # D_soft -> c D_soft. log D_soft -> log D_soft + log c.
        # Term: log_prod - 28 log D_soft.
        # Change: 28 log c - 28 log c = 0.
        # So it is scale invariant. Good.
        
        # To prevent overflow in d^k, we can subtract the max log d?
        # log(sum(d^k)) = k * log(max d) + log(sum((d/max d)^k))
        # This is more stable.
        max_d = np.max(d)
        if max_d < 1e-10:
            return -1e10
        scaled_d = d / max_d
        log_sum_exp = k * np.log(max_d) + np.log(np.sum(scaled_d ** k))
        log_D_soft = log_sum_exp / k
        
        return log_prod - 28 * log_D_soft

    def neg_smooth_score(x):
        return -log_score(x)

    best_score = -np.inf
    best_x = x0
    
    # Try the regular octagon
    s = log_score(x0)
    if s > best_score:
        best_score = s
        best_x = x0.copy()
    
    # Optimization attempts
    # 1. From regular octagon
    try:
        res = minimize(neg_smooth_score, x0, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-12})
        if res.success:
            s = log_score(res.x)
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
    except Exception:
        pass
    
    # 2. Random perturbations
    np.random.seed(42)
    for i in range(20):
        if time.time() > deadline:
            break
            
        # Small perturbation
        noise = np.random.normal(0, 0.1, size=x0.shape)
        x_init = x0 + noise
        
        try:
            res = minimize(neg_smooth_score, x_init, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-10})
            if res.success:
                s = log_score(res.x)
                if s > best_score:
                    best_score = s
                    best_x = res.x.copy()
        except Exception:
            pass

    # 3. Try Nelder-Mead on the best found so far for polish
    # Nelder-Mead doesn't use gradients, might be better for the non-smooth k->inf limit
    # But L-BFGS with k=50 should be close.
    # Let's try one NM run on the best point found.
    try:
        res = minimize(neg_smooth_score, best_x, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.success:
            s = log_score(res.x)
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
    except Exception:
        pass

    # Convert back to list of lists
    best_pts = best_x.reshape(n, 2)
    
    # Ensure distinctness and validity
    # If any points are too close, nudge them
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-6:
                best_pts[j] += np.random.normal(0, 0.01, 2)

    return best_pts.tolist()
