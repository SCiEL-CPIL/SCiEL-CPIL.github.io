# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    timeout = 600  # Safety margin before 630s limit
    
    # Initial guess: 4x4 square grid
    # This is a known strong baseline for N=16
    x = np.linspace(0, 3, 4)
    y = np.linspace(0, 3, 4)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack((X.ravel(), Y.ravel()))
    
    # Center the points to remove translational invariance
    points -= points.mean(axis=0)
    
    # Scale to unit max distance to avoid numerical issues? 
    # The objective is scale invariant, so we can fix scale if needed, 
    # but L-BFGS-B handles it fine if we just maximize the ratio.
    
    # We want to maximize R = (d_min / d_max)^2
    # Equivalent to minimizing -log(R) = -2 * (log(d_min) - log(d_max))
    # Or minimizing log(d_max) - log(d_min)
    
    # We use a soft approximation for log(d_min) and log(d_max)
    # log(d_max) approx (1/beta) * log(sum(d^beta))
    # log(d_min) approx -(1/beta) * log(sum(d^-beta))
    
    beta = 20.0
    
    def objective(flat_points):
        pts = flat_points.reshape(n, d)
        dists = pdist(pts)
        
        # Handle zeros or very small distances to avoid log(0) or inf
        # Add a small epsilon
        eps = 1e-12
        dists = dists + eps
        
        # Soft max log
        # sum(exp(beta * log(d))) = sum(d^beta)
        # We want log(sum(d^beta)) / beta
        # To avoid overflow, we can use the log-sum-exp trick
        log_d = np.log(dists)
        
        # Soft max: log(sum(exp(beta * log_d))) / beta
        # Let M = max(beta * log_d)
        # log(sum(exp(x))) = M + log(sum(exp(x - M)))
        x_max = beta * log_d
        M_max = np.max(x_max)
        soft_max_log = (M_max + np.log(np.sum(np.exp(x_max - M_max)))) / beta
        
        # Soft min: -log(sum(exp(-beta * log_d))) / beta
        x_min = -beta * log_d
        M_min = np.max(x_min)
        soft_min_log = -(M_min + np.log(np.sum(np.exp(x_min - M_min)))) / beta
        
        # Objective to minimize: log(d_max) - log(d_min)
        return soft_max_log - soft_min_log

    # Convert to flat vector for optimizer
    x0 = points.flatten()
    
    best_points = points.copy()
    best_score = objective(x0)
    
    # Optimization loop with time check
    # We can run multiple optimizations with different betas or just one robust one
    # L-BFGS-B is good for this
    
    def callback(xk):
        nonlocal best_points, best_score
        curr_score = objective(xk)
        if curr_score < best_score:
            best_score = curr_score
            best_points = xk.reshape(n, d)
        
        if time.time() - start_time > timeout:
            raise Exception("Timeout")
        
        return False

    try:
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            callback=callback,
            options={'maxiter': 10000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < best_score:
            best_points = res.x.reshape(n, d)
            
    except Exception as e:
        # If timeout or other error, use best_points found so far
        pass
    
    # Final polish with higher beta for better precision if time allows
    if time.time() - start_time < timeout - 10:
        beta_final = 50.0
        
        def objective_final(flat_points):
            pts = flat_points.reshape(n, d)
            dists = pdist(pts)
            eps = 1e-12
            dists = dists + eps
            log_d = np.log(dists)
            
            x_max = beta_final * log_d
            M_max = np.max(x_max)
            soft_max_log = (M_max + np.log(np.sum(np.exp(x_max - M_max)))) / beta_final
            
            x_min = -beta_final * log_d
            M_min = np.max(x_min)
            soft_min_log = -(M_min + np.log(np.sum(np.exp(x_min - M_min)))) / beta_final
            
            return soft_max_log - soft_min_log

        x0_final = best_points.flatten()
        best_points_final = best_points.copy()
        best_score_final = objective_final(x0_final)
        
        def callback_final(xk):
            nonlocal best_points_final, best_score_final
            curr_score = objective_final(xk)
            if curr_score < best_score_final:
                best_score_final = curr_score
                best_points_final = xk.reshape(n, d)
            if time.time() - start_time > timeout:
                raise Exception("Timeout")
            return False

        try:
            res_final = minimize(
                objective_final, 
                x0_final, 
                method='L-BFGS-B', 
                callback=callback_final,
                options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_final.success or res_final.fun < best_score_final:
                best_points_final = res_final.x.reshape(n, d)
        except Exception:
            pass
        
        best_points = best_points_final

    return best_points

# EVOLVE-BLOCK-END
