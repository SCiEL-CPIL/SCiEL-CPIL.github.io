"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def get_sq_dists(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        d_sq = get_sq_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if scale > 1e-9:
            pts = pts / scale
        return pts

    # Strategy: Gradient-based optimization (L-BFGS-B) on a smooth surrogate.
    # The raw objective (max/min ratio) is non-smooth. We use a soft-max approximation
    # to create a differentiable landscape that L-BFGS-B can navigate efficiently.
    # We optimize log(sum(exp(d_i / T))) which approximates log(max(d_i)) as T -> 0.
    # Minimizing this encourages all distances to be small and similar, indirectly
    # minimizing the ratio by preventing any distance from becoming an outlier.
    # We run multiple restarts with different initializations and decreasing T.
    
    def soft_max_obj(x, T=1.0):
        pts = x.reshape(N, 2)
        d_sq = get_sq_dists(pts)
        # Avoid log(0)
        d_sq = np.maximum(d_sq, 1e-10)
        # Use log-sum-exp for stability
        d = np.sqrt(d_sq)
        m = np.max(d)
        log_sum_exp = m + np.log(np.sum(np.exp((d - m) / T)))
        # We want to minimize the spread. 
        # A common proxy for min-ratio is minimizing the variance of log-distances
        # or maximizing the min-distance subject to max-distance constraint.
        # Here, we minimize the "soft max" distance. This pulls all points closer to center.
        # To balance min vs max, we can minimize: soft_max(d) - lambda * soft_min(d)
        # Or simply minimize the range: soft_max(d) - soft_min(d)
        
        # Let's try minimizing the ratio directly using smooth approximations.
        # max_d ~ T * log(sum(exp(d/T)))
        # min_d ~ -T * log(sum(exp(-d/T)))
        # ratio ~ max_d / min_d
        # This is still non-smooth at min_d=0.
        
        # Better approach: Minimize the variance of the log-distances?
        # Or minimize: sum(d_i^2) - c * min(d_i)^2 ?
        
        # Let's use the objective: f(x) = log(sum(exp(d_i/T))) - log(sum(exp(-d_i/T)))
        # This approximates max(d) - min(d) = range. Minimizing range helps ratio.
        
        neg_d = -d
        m_neg = np.max(neg_d)
        log_sum_exp_neg = m_neg + np.log(np.sum(np.exp((neg_d - m_neg) / T)))
        
        # Objective: soft_max - soft_min
        return log_sum_exp - log_sum_exp_neg

    def soft_max_grad(x, T=1.0):
        pts = x.reshape(N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = sq_dists[iu]
        d_sq = np.maximum(d_sq, 1e-10)
        d = np.sqrt(d_sq)
        
        # Gradient of soft_max(d) w.r.t d
        # d/d_i = exp(d_i/T) / sum(exp(d_j/T))
        m = np.max(d)
        exp_d = np.exp((d - m) / T)
        sum_exp_d = np.sum(exp_d)
        w_max = exp_d / sum_exp_d
        
        # Gradient of soft_min(d) = -soft_max(-d)
        # d/d_i = -exp(-d_i/T) / sum(exp(-d_j/T))
        m_neg = np.max(-d)
        exp_neg_d = np.exp((-d - m_neg) / T)
        sum_exp_neg_d = np.sum(exp_neg_d)
        w_min = exp_neg_d / sum_exp_neg_d
        
        # Objective is soft_max - soft_min
        # Grad obj = w_max - w_min
        weights = w_max - w_min
        
        # Now backpropagate to points
        # d/dx_k (d_ij) = 2 * (x_i - x_j) for i != k, -2(x_j - x_i) for j = k ...
        # Actually, d/dx_k (||x_i - x_j||^2) = 2(x_i - x_j) if k=i, -2(x_i-x_j) if k=j, 0 otherwise.
        # d/dx_k d_ij = (d/dx_k d_ij^2) / (2 d_ij)
        
        grad = np.zeros_like(pts)
        for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
            w = weights[idx]
            if d[idx] < 1e-8:
                continue
            diff_vec = pts[i] - pts[j]
            grad[i] += w * diff_vec / d[idx]
            grad[j] -= w * diff_vec / d[idx]
            
        return grad.ravel()

    # Initial configurations
    initial_configs = []
    
    # 1. Regular Grid (4x4) - good starting point for "equidistant" structures
    x = np.arange(4) - 1.5
    y = np.arange(4) - 1.5
    xx, yy = np.meshgrid(x, y)
    base_grid = np.column_stack([xx.ravel(), yy.ravel()])
    initial_configs.append(base_grid.copy())
    
    # 2. Perturbed Grid
    noise = np.random.normal(0, 0.5, (N, 2))
    initial_configs.append(base_grid + noise)
    
    # 3. Random
    initial_configs.append(np.random.uniform(-2, 2, (N, 2)))
    
    # 4. Ring (8 inner, 8 outer)
    angles_inner = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles_outer = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    inner_pts = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    outer_pts = np.column_stack([1.0 * np.cos(angles_outer), 1.0 * np.sin(angles_outer)])
    initial_configs.append(np.vstack([inner_pts, outer_pts]))
    
    best_overall = None
    best_overall_score = 1e10
    
    for init_pts in initial_configs:
        if time.time() > deadline:
            break
            
        init_pts = normalize(init_pts)
        x0 = init_pts.ravel()
        
        # Multi-stage optimization: Start with high T (smooth), lower T (sharper)
        temperatures = [1.0, 0.5, 0.2, 0.1]
        current_x = x0
        
        for T in temperatures:
            if time.time() > deadline:
                break
                
            try:
                res = minimize(
                    soft_max_obj,
                    current_x,
                    args=(T,),
                    jac=soft_max_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < 1e10: # Check for failure
                    current_x = res.x
            except Exception:
                pass
        
        # Evaluate final score
        final_pts = normalize(current_x.reshape(N, 2))
        final_score = score_raw(final_pts)
        
        if final_score < best_overall_score:
            best_overall_score = final_score
            best_overall = final_pts

    # Final local polish with Nelder-Mead on the raw score if time permits
    if time.time() < deadline - 10 and best_overall is not None:
        def raw_obj(x):
            pts = x.reshape(N, 2)
            return score_raw(pts)
        
        try:
            res = minimize(
                raw_obj,
                best_overall.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < best_overall_score:
                best_overall = res.x.reshape(N, 2)
                best_overall_score = res.fun
        except Exception:
            pass

    return best_overall

# EVOLVE-BLOCK-END
