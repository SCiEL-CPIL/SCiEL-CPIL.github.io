import numpy as np
from scipy.optimize import minimize
import time

N = 14
D = 3

def _pairwise_sq_dists(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(N, k=1)
    return dist_sq[iu]

def _objective(coords, t):
    points = coords.reshape(N, D)
    # Center the points to remove translation invariance
    points = points - np.mean(points, axis=0)
    
    dist_sq = _pairwise_sq_dists(points)
    
    max_ds = np.max(dist_sq)
    min_ds = np.min(dist_sq)
    
    if max_ds < 1e-10:
        return 1e10
        
    # Soft max/min
    # Use np.logaddexp.reduce for numerical stability if needed, but sum(exp) is usually fine for t<1000
    # To be safe, we can use log-sum-exp trick
    # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    
    x = dist_sq
    max_x = np.max(x)
    log_sum_exp_max = max_x + np.log(np.sum(np.exp(t * (x - max_x))))
    log_soft_max = log_sum_exp_max / t
    
    min_x = np.min(x)
    # For min, we use -log_sum_exp(-x)
    # log(sum(exp(-t*x))) = min(-t*x) + log(sum(exp(-t*x - min(-t*x))))
    # min(-t*x) = -t*max(x)
    # This is getting complicated. Let's stick to the previous formulation but ensure stability.
    
    # Previous formulation:
    # log_soft_max = max_ds + np.log(np.sum(np.exp(t * (dist_sq - max_ds)))) / t
    # log_soft_min = min_ds - np.log(np.sum(np.exp(-t * (dist_sq - min_ds)))) / t
    
    # This is stable as long as t * (dist_sq - max_ds) is not too large negative (underflow to 0 is fine).
    # t * (dist_sq - min_ds) is not too large positive (overflow).
    # dist_sq - min_ds >= 0. If dist_sq is large, exp can overflow.
    # But dist_sq is bounded by the initialization and optimization.
    # To be safe, we can clip or use a smaller max t.
    
    exp_term_max = np.exp(t * (dist_sq - max_ds))
    log_soft_max = max_ds + np.log(np.sum(exp_term_max)) / t
    
    exp_term_min = np.exp(-t * (dist_sq - min_ds))
    log_soft_min = min_ds - np.log(np.sum(exp_term_min)) / t
    
    return log_soft_max - log_soft_min

def construct():
    start_time = time.time()
    deadline = 115.0
    
    best_score = np.inf
    best_points = None
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        points = rng.random((N, D)) * 2 - 1
        points /= (np.linalg.norm(points, axis=1, keepdims=True) + 1e-6)
        
        current_coords = points.flatten()
        
        t_schedule = [1.0, 10.0, 100.0, 1000.0]
        
        for t in t_schedule:
            if time.time() > deadline:
                break
                
            try:
                res = minimize(
                    _objective, 
                    current_coords, 
                    args=(t,), 
                    method='L-BFGS-B', 
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                # Use res.x even if not success, as it might be better
                current_coords = res.x
            except Exception:
                pass
        
        final_points = current_coords.reshape(N, D)
        # Center for final score calculation? No, score is invariant.
        dist_sq = _pairwise_sq_dists(final_points)
        min_d = np.min(dist_sq)
        max_d = np.max(dist_sq)
        
        if min_d < 1e-10:
            score = 1e10
        else:
            score = max_d / min_d
            
        if score < best_score:
            best_score = score
            best_points = final_points.copy()

    if best_points is None:
        side = int(np.ceil(N ** (1 / D)))
        grid = np.array([[i, j, k] for i in range(side) for j in range(side) for k in range(side)][:N], dtype=float)
        return grid
        
    return best_points
