"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # 1. Generate a high-quality initial configuration
    # For N=16, a 4x4 square grid is a strong candidate for the global optimum
    # of the distance ratio problem, as it maximizes the minimum distance 
    # relative to the maximum distance for a square-like container.
    # We start with a 4x4 grid centered at the origin.
    
    side = 4
    x = np.arange(side) - (side - 1) / 2.0
    y = np.arange(side) - (side - 1) / 2.0
    xx, yy = np.meshgrid(x, y)
    init_pts = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Normalize to unit variance to stabilize optimization
    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if scale > 1e-9:
            pts = pts / scale
        return pts

    current_pts = normalize(init_pts)

    # 2. Define the smooth objective function
    # We minimize (max_d / min_d)^2 = max_d^2 / min_d^2
    # Let u = log(min_d^2) and v = log(max_d^2).
    # We want to minimize exp(v - u) = exp(v) * exp(-u).
    # However, scipy minimize doesn't handle log-softmax well directly if we just use logs.
    # A better smooth surrogate for the ratio is:
    # f(pts) = (1/N_pairs) * sum(exp(alpha * d_ij^2)) / (1/N_pairs) * sum(exp(-alpha * d_ij^2))
    # As alpha -> infinity, this approaches max(d^2) / min(d^2).
    # But this can be numerically unstable.
    
    # Alternative smooth approach:
    # Minimize J(pts) = max_i d_i^2 - lambda * min_i d_i^2 ? No, we want ratio.
    
    # Let's stick to the log-sum-exp trick for a smooth approximation of the ratio.
    # Let D^2 be the vector of squared distances.
    # log(max D^2) ≈ (1/alpha) * log(sum(exp(alpha * D^2 / 2))) ? No.
    # log(max x) ≈ (1/alpha) log(sum(exp(alpha x))).
    # log(min x) = -log(max (1/x)) ≈ -(1/alpha) log(sum(exp(alpha / x))).
    
    # So, log_ratio = log(max D^2) - log(min D^2)
    # ≈ (1/alpha) [ log(sum(exp(alpha D^2))) - log(sum(exp(alpha / D^2))) ]
    # We want to minimize exp(2 * log_ratio) = ratio^2.
    # Let's minimize log_ratio^2 directly? Or just log_ratio?
    # Minimizing log_ratio is equivalent to minimizing ratio.
    
    def get_sq_dists(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def smooth_objective(x, alpha=10.0):
        # x is flattened points
        pts = x.reshape(N, 2)
        d_sq = get_sq_dists(pts)
        
        # Avoid division by zero or negative values (shouldn't happen with distinct points)
        # d_sq should be > 0.
        if np.any(d_sq < 1e-12):
            return 1e10
            
        # log(max d_sq) approx
        # To prevent overflow in exp(alpha * d_sq), we can subtract the max first?
        # But we need the sum of exps.
        # Standard log-sum-exp: log(sum(exp(a_i))) = max(a_i) + log(sum(exp(a_i - max(a_i))))
        
        a_max = alpha * d_sq
        a_min = alpha / d_sq  # For min distance term
        
        # log(max d_sq) = (1/alpha) * (max(a_max) + log(sum(exp(a_max - max(a_max)))))
        m_max = np.max(a_max)
        log_max_d_sq = (1.0 / alpha) * (m_max + np.log(np.sum(np.exp(a_max - m_max))))
        
        # log(min d_sq) = - (1/alpha) * (max(a_min) + log(sum(exp(a_min - max(a_min)))))
        m_min = np.max(a_min)
        log_min_d_sq = - (1.0 / alpha) * (m_min + np.log(np.sum(np.exp(a_min - m_min))))
        
        log_ratio = log_max_d_sq - log_min_d_sq
        # We want to minimize ratio^2 = exp(2 * log_ratio)
        # Minimizing log_ratio is sufficient and more stable.
        return log_ratio

    # Gradient of the smooth objective is not easily derivable by hand in code quickly,
    # so we use numerical gradient or just let L-BFGS-B estimate it.
    # However, L-BFGS-B is very efficient with numerical gradients for small N.
    
    # Let's try a different, simpler smooth objective that is differentiable everywhere:
    # f(pts) = (sum d_ij^2)^2 / (prod d_ij^2) ? No.
    
    # Let's use the "soft-min" and "soft-max" on the distances themselves, not squared.
    # Or better, use the actual ratio but with a smooth max/min approximation.
    
    # Actually, for N=16, the landscape is likely simple enough that a direct
    # optimization of the non-smooth function with a very good initial guess
    # and a specialized solver might work, but the directive asks for a structural change.
    
    # Let's implement the log-sum-exp based objective with gradient.
    # Gradient of log(sum(exp(alpha * d_sq))) wrt pts:
    # d/dx log(sum(exp(alpha * d_sq))) = (sum(exp(alpha * d_sq) * alpha * 2(d_i - d_j) * e_k)) / sum(exp(alpha * d_sq))
    # where e_k is the unit vector for point k.
    
    def smooth_objective_and_grad(x, alpha=20.0):
        pts = x.reshape(N, 2)
        d_sq = get_sq_dists(pts)
        if np.any(d_sq < 1e-12):
            return 1e10, np.zeros_like(x)
            
        # Terms for max
        a_max = alpha * d_sq
        m_max = np.max(a_max)
        exp_a_max = np.exp(a_max - m_max)
        sum_exp_max = np.sum(exp_a_max)
        
        # Terms for min (using 1/d_sq)
        a_min = alpha / d_sq
        m_min = np.max(a_min)
        exp_a_min = np.exp(a_min - m_min)
        sum_exp_min = np.sum(exp_a_min)
        
        log_max_d_sq = (1.0 / alpha) * (m_max + np.log(sum_exp_max))
        log_min_d_sq = - (1.0 / alpha) * (m_min + np.log(sum_exp_min))
        
        val = log_max_d_sq - log_min_d_sq
        
        # Gradient calculation
        grad = np.zeros_like(x)
        
        # d(log_max_d_sq)/dx
        # = (1/alpha) * [ (1/sum_exp_max) * sum( exp(a_max_k - m_max) * alpha * 2 * (p_i - p_j) * e_i ) ]
        # Note: d(d_sq_k)/dx = 2(p_i - p_j) for point i, -2(p_i - p_j) for point j, 0 otherwise.
        
        weights_max = exp_a_max / sum_exp_max
        weights_min = exp_a_min / sum_exp_min
        
        # Iterate over pairs to accumulate gradient
        iu = np.triu_indices(N, 1)
        i_idx, j_idx = iu
        
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            diff = pts[i] - pts[j]
            
            # Contribution to max term
            # d log_max / dx_i += (1/alpha) * w_k * alpha * 2 * diff = 2 * w_k * diff
            # d log_max / dx_j += (1/alpha) * w_k * alpha * (-2) * diff = -2 * w_k * diff
            grad[2*i:2*i+2] += 2.0 * weights_max[k] * diff
            grad[2*j:2*j+2] -= 2.0 * weights_max[k] * diff
            
            # Contribution to min term
            # log_min_d_sq = - (1/alpha) * (m_min + log(sum_exp_min))
            # d log_min / dx = - (1/alpha) * (1/sum_exp_min) * sum( exp(a_min_k - m_min) * d(a_min_k)/dx )
            # a_min_k = alpha / d_sq_k
            # d(a_min_k)/dx_i = alpha * d(1/d_sq_k)/dx_i = alpha * (-1/d_sq_k^2) * d(d_sq_k)/dx_i
            # d(d_sq_k)/dx_i = 2 * diff
            # So d(a_min_k)/dx_i = -2 * alpha * diff / d_sq_k^2
            # d log_min / dx_i = - (1/alpha) * w_k * (-2 * alpha * diff / d_sq_k^2) = 2 * w_k * diff / d_sq_k^2
            # d log_min / dx_j = -2 * w_k * diff / d_sq_k^2
            
            d_sq_k = d_sq[k]
            term_min = 2.0 * weights_min[k] * diff / (d_sq_k * d_sq_k)
            
            # val = log_max - log_min
            # d(val)/dx_i = d(log_max)/dx_i - d(log_min)/dx_i
            grad[2*i:2*i+2] -= term_min
            grad[2*j:2*j+2] += term_min
            
        return val, grad

    # Optimization
    best_x = current_pts.ravel()
    best_score = 1e10
    
    def raw_score(pts):
        d_sq = get_sq_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2
    
    current_score = raw_score(current_pts)
    if current_score < best_score:
        best_score = current_score
        best_x = current_pts.ravel().copy()
        
    # Run L-BFGS-B
    # We might need to run it multiple times with different alphas (annealing alpha)
    # Start with small alpha (smoother) and increase.
    
    alphas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    for alpha in alphas:
        if time.time() > deadline:
            break
            
        res = minimize(
            smooth_objective_and_grad,
            best_x,
            args=(alpha,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
        )
        
        if res.success or res.fun < smooth_objective_and_grad(best_x, alpha)[0]:
            new_pts = res.x.reshape(N, 2)
            new_score = raw_score(new_pts)
            if new_score < best_score:
                best_score = new_score
                best_x = res.x.copy()
                current_pts = new_pts

    # Final check and return
    final_pts = best_x.reshape(N, 2)
    
    # One last local polish with a different method? 
    # Maybe just return the best found.
    
    return final_pts

# EVOLVE-BLOCK-END
