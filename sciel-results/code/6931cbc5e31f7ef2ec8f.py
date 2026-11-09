import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2
    start_time = time.time()
    time_limit = 600  # seconds
    
    # Initial guess: points on a circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Flatten points for scipy optimizer
    x0 = points0.flatten()
    
    def objective_and_grad(x):
        points = x.reshape(n, 2)
        # Compute pairwise distances
        # Use pdist for efficiency
        dists = pdist(points)
        
        # We need weights for soft-min and soft-max
        # Soft-min approx: -1/beta * log(sum(exp(-beta * d)))
        # Soft-max approx: 1/beta * log(sum(exp(beta * d)))
        # Objective to maximize: log(d_min) - log(d_max)
        # Surrogate: -1/beta * log(sum(exp(-beta*d))) - 1/beta * log(sum(exp(beta*d)))
        
        # To avoid overflow/underflow, we can shift the exponents
        # But for moderate beta, it's usually fine.
        # Let's use beta = 10 for stability, maybe increase later.
        beta = 10.0
        
        # Calculate terms
        # exp(-beta * d)
        # d is positive.
        # If d is very large, exp(-beta*d) is 0.
        # If d is very small, exp(-beta*d) is 1.
        
        # For numerical stability, we can use log-sum-exp
        # log(sum(exp(A))) = max(A) + log(sum(exp(A - max(A))))
        
        A1 = -beta * dists
        A2 = beta * dists
        
        max_A1 = np.max(A1)
        max_A2 = np.max(A2)
        
        log_S1 = max_A1 + np.log(np.sum(np.exp(A1 - max_A1)))
        log_S2 = max_A2 + np.log(np.sum(np.exp(A2 - max_A2)))
        
        val = -1.0 / beta * (log_S1 + log_S2)
        
        # Gradient
        # w1_k = exp(A1_k) / sum(exp(A1)) = exp(A1_k - log_S1)
        # w2_k = exp(A2_k) / sum(exp(A2)) = exp(A2_k - log_S2)
        
        w1 = np.exp(A1 - log_S1)
        w2 = np.exp(A2 - log_S2)
        
        # grad_obj = sum_k (w1_k - w2_k) * grad_d_k
        # We need to accumulate gradients for each point.
        grad = np.zeros_like(points)
        
        # dists is from pdist, which is condensed form.
        # We need to map back to pairs (i, j) with i < j.
        # Number of pairs is n*(n-1)/2.
        
        # Efficiently compute gradient contributions
        # For each pair (i, j):
        # contrib = (w1[k] - w2[k]) / d_k * (P_i - P_j)
        # grad[P_i] += contrib
        # grad[P_j] -= contrib
        
        # pdist order: (0,1), (0,2), ..., (0,n-1), (1,2), ...
        
        # Vectorized approach for gradient accumulation is tricky with pdist.
        # Let's use a loop over pairs if n is small (16 -> 120 pairs).
        # 120 iterations is fast.
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                d_k = np.linalg.norm(diff)
                if d_k < 1e-10:
                    # Avoid division by zero
                    grad_val = 0
                else:
                    # w1[idx] - w2[idx]
                    coeff = (w1[idx] - w2[idx]) / d_k
                    grad_vec = coeff * diff
                    grad[i] += grad_vec
                    grad[j] -= grad_vec
                idx += 1
        
        # We want to maximize val, so minimize -val
        return -val, -grad.flatten()

    # Run L-BFGS-B
    # We might need multiple restarts or different betas.
    # Let's try one long run first.
    
    best_points = points0
    best_score = -np.inf
    
    # Helper to calculate actual score
    def calc_score(pts):
        if np.all(pts == 0):
            return 0.0
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Initial score
    best_score = calc_score(points0)
    
    # Optimization loop
    # We can try different betas.
    # Start with beta=10.
    
    beta_vals = [10.0, 20.0, 50.0]
    
    for beta in beta_vals:
        if time.time() - start_time > time_limit - 30:
            break
            
        # Redefine objective with specific beta
        def obj(x, b=beta):
            points = x.reshape(n, 2)
            dists = pdist(points)
            
            A1 = -b * dists
            A2 = b * dists
            
            max_A1 = np.max(A1)
            max_A2 = np.max(A2)
            
            log_S1 = max_A1 + np.log(np.sum(np.exp(A1 - max_A1)))
            log_S2 = max_A2 + np.log(np.sum(np.exp(A2 - max_A2)))
            
            val = -1.0 / b * (log_S1 + log_S2)
            
            w1 = np.exp(A1 - log_S1)
            w2 = np.exp(A2 - log_S2)
            
            grad = np.zeros_like(points)
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    diff = points[i] - points[j]
                    d_k = np.linalg.norm(diff)
                    if d_k < 1e-10:
                        coeff = 0
                    else:
                        coeff = (w1[idx] - w2[idx]) / d_k
                        grad_vec = coeff * diff
                        grad[i] += grad_vec
                        grad[j] -= grad_vec
                    idx += 1
            
            return -val, -grad.flatten()

        res = minimize(obj, x0, method='L-BFGS-B', jac=True, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success:
            pts_opt = res.x.reshape(n, 2)
            score = calc_score(pts_opt)
            if score > best_score:
                best_score = score
                best_points = pts_opt
                x0 = res.x # Use best result as start for next beta
        
    # Polish with Nelder-Mead on the true objective?
    # Nelder-Mead doesn't use gradients, so it can handle non-smoothness.
    # But it's slow. Let's try a few iterations if time permits.
    
    if time.time() - start_time < time_limit - 10:
        # Define true objective for Nelder-Mead (to minimize negative score)
        def true_obj(x):
            pts = x.reshape(n, 2)
            return -calc_score(pts)
        
        # Use best_points as start
        x0_nm = best_points.flatten()
        # Nelder-Mead might be slow, so limit iterations
        try:
            res_nm = minimize(true_obj, x0_nm, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res_nm.success:
                pts_nm = res_nm.x.reshape(n, 2)
                score_nm = calc_score(pts_nm)
                if score_nm > best_score:
                    best_score = score_nm
                    best_points = pts_nm
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
