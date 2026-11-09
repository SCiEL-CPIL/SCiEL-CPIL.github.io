"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

# We will use a log-sum-exp style smooth surrogate for the diameter constraint,
# and optimize the 8 points directly in R^16 (or with symmetry breaking).
# The objective is to maximize the sum of log(distances) for all ordered pairs,
# subject to the diameter being <= 2.
#
# The score is Delta / n^n where Delta is the product of distances over ordered pairs
# after scaling to diameter 2.
# Let D be the set of 8 points. Let d_max = max_{i != j} ||D_i - D_j||.
# Scale D to D' = D / d_max * 2.
# Then the product of distances over ordered pairs for D' is:
#   P' = prod_{i != j} ||D'_i - D'_j|| = prod_{i != j} (2/d_max) * ||D_i - D_j||
#      = (2/d_max)^(n(n-1)) * prod_{i != j} ||D_i - D_j||
#      = (2/d_max)^(n(n-1)) * Delta_raw
#
# The score is P' / n^n.
# Since n=8, n(n-1) = 56.
# Score = (2/d_max)^56 * Delta_raw / 8^8.
#
# To maximize this, we can just maximize log(Delta_raw) - 56 * log(d_max) + const.
# This is equivalent to maximizing log(Delta_raw) - 56 * log(max_{i!=j} ||D_i - D_j||).
#
# The term log(max_{i!=j} ||D_i - D_j||) is non-smooth. We can use a soft-max:
#   log_max_{i!=j} (||D_i - D_j||) ≈ (1/T) * log( sum_{i!=j} exp(T * ||D_i - D_j||) )
# as T -> infinity, this approaches the max.
#
# So the objective to maximize is:
#   F(D) = sum_{i!=j} log(||D_i - D_j||) - 56 * (1/T) * log( sum_{i!=j} exp(T * ||D_i - D_j||) )
#
# We can use scipy.optimize.minimize on -F(D).

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 8
    # Regular octagon as initial guess
    angles = 2 * np.pi * np.arange(n) / n
    x0 = np.concatenate([np.cos(angles), np.sin(angles)])
    
    def objective(x):
        pts = x.reshape(n, 2)
        # Compute all pairwise distances
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        # Mask out diagonal
        mask = ~np.eye(n, dtype=bool)
        dists_masked = dists[mask]
        
        # Sum of log distances
        # Avoid log(0) by clipping
        dists_masked_clipped = np.maximum(dists_masked, 1e-15)
        sum_log = np.sum(np.log(dists_masked_clipped))
        
        # Soft-max of distances
        T = 50.0
        # For numerical stability, subtract max
        d_max_val = np.max(dists_masked_clipped)
        log_sum_exp = d_max_val + np.log(np.sum(np.exp(T * (dists_masked_clipped - d_max_val)))) / T
        # Actually, log_sum_exp as defined above is (1/T) * log(sum(exp(T*d)))
        # Let's recompute:
        # log_sum_exp = (1/T) * log(sum(exp(T * dists)))
        # = (1/T) * (T * d_max + log(sum(exp(T * (dists - d_max)))))
        # = d_max + (1/T) * log(sum(exp(T * (dists - d_max))))
        
        # Objective to maximize: sum_log - 56 * log_sum_exp
        # We return the negative for minimization
        return -(sum_log - 56.0 * log_sum_exp)
    
    def objective_grad(x):
        pts = x.reshape(n, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        mask = ~np.eye(n, dtype=bool)
        dists_masked = dists[mask]
        dists_masked_clipped = np.maximum(dists_masked, 1e-15)
        
        # Gradient of sum_log: sum_{i!=j} 1/||D_i - D_j|| * grad_{D_i, D_j} ||D_i - D_j||
        # grad_{D_k} sum_log = sum_{j != k} (D_k - D_j) / ||D_k - D_j||^2
        # Actually, d/dx_k log(||x_i - x_j||) = (x_k - x_j)/||x_i - x_j||^2 if k=i, (x_j - x_i)/||x_i - x_j||^2 if k=j
        # So grad_{x_k} sum_{i!=j} log(||x_i - x_j||) = sum_{j != k} (x_k - x_j)/||x_k - x_j||^2
        
        grad_sum_log = np.zeros_like(pts)
        for k in range(n):
            for j in range(n):
                if k != j:
                    d = pts[k] - pts[j]
                    d_norm = np.sqrt(np.sum(d**2))
                    if d_norm > 1e-15:
                        grad_sum_log[k] += d / (d_norm**2)
        
        # Gradient of soft-max term:
        # g = (1/T) log(sum_{i!=j} exp(T * d_{ij}))
        # grad_{x_k} g = (1/T) * sum_{i!=j} T * exp(T*d_{ij})/sum(exp(T*d_{ij})) * grad_{x_k} d_{ij}
        #               = sum_{i!=j} p_{ij} * grad_{x_k} d_{ij}
        # where p_{ij} = exp(T*d_{ij}) / sum_{a!=b} exp(T*d_{ab})
        # grad_{x_k} d_{ij} = (x_k - x_j)/||x_i - x_j|| if k=i, (x_j - x_i)/||x_i - x_j|| if k=j, 0 otherwise
        
        T = 50.0
        d_max_val = np.max(dists_masked_clipped)
        exp_vals = np.exp(T * (dists_masked_clipped - d_max_val))
        sum_exp = np.sum(exp_vals)
        probs = exp_vals / sum_exp
        
        grad_softmax = np.zeros_like(pts)
        idx = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    p = probs[idx]
                    if p > 1e-15:
                        d = pts[i] - pts[j]
                        d_norm = np.sqrt(np.sum(d**2))
                        if d_norm > 1e-15:
                            grad_softmax[i] += p * d / d_norm
                            grad_softmax[j] -= p * d / d_norm
                    idx += 1
        
        # Total gradient of objective (to maximize): grad_sum_log - 56 * grad_softmax
        # We return negative for minimization
        total_grad = grad_sum_log - 56.0 * grad_softmax
        return -total_grad.flatten()
    
    # Try a few different initial guesses and T values
    best_x = x0
    best_val = objective(x0)
    
    # Regular octagon
    initial_guesses = [x0]
    
    # Slightly perturbed regular octagon
    rng = np.random.default_rng(42)
    for _ in range(5):
        x_pert = x0 + rng.normal(0, 0.01, size=16)
        initial_guesses.append(x_pert)
    
    # Different T values
    T_values = [20.0, 50.0, 100.0]
    
    for T in T_values:
        if time.time() > deadline:
            break
        
        # Redefine objective and grad with this T
        def obj_T(x, T=T):
            pts = x.reshape(n, 2)
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            mask = ~np.eye(n, dtype=bool)
            dists_masked = dists[mask]
            dists_masked_clipped = np.maximum(dists_masked, 1e-15)
            sum_log = np.sum(np.log(dists_masked_clipped))
            d_max_val = np.max(dists_masked_clipped)
            log_sum_exp = d_max_val + np.log(np.sum(np.exp(T * (dists_masked_clipped - d_max_val)))) / T
            return -(sum_log - 56.0 * log_sum_exp)
        
        def grad_T(x, T=T):
            pts = x.reshape(n, 2)
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            mask = ~np.eye(n, dtype=bool)
            dists_masked = dists[mask]
            dists_masked_clipped = np.maximum(dists_masked, 1e-15)
            
            grad_sum_log = np.zeros_like(pts)
            for k in range(n):
                for j in range(n):
                    if k != j:
                        d = pts[k] - pts[j]
                        d_norm = np.sqrt(np.sum(d**2))
                        if d_norm > 1e-15:
                            grad_sum_log[k] += d / (d_norm**2)
            
            d_max_val = np.max(dists_masked_clipped)
            exp_vals = np.exp(T * (dists_masked_clipped - d_max_val))
            sum_exp = np.sum(exp_vals)
            probs = exp_vals / sum_exp
            
            grad_softmax = np.zeros_like(pts)
            idx = 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        p = probs[idx]
                        if p > 1e-15:
                            d = pts[i] - pts[j]
                            d_norm = np.sqrt(np.sum(d**2))
                            if d_norm > 1e-15:
                                grad_softmax[i] += p * d / d_norm
                                grad_softmax[j] -= p * d / d_norm
                        idx += 1
            
            total_grad = grad_sum_log - 56.0 * grad_softmax
            return -total_grad.flatten()
        
        for x_init in initial_guesses:
            if time.time() > deadline:
                break
            
            try:
                from scipy.optimize import minimize
                result = minimize(obj_T, x_init, jac=grad_T, method='BFGS', 
                                  options={'maxiter': 500, 'gtol': 1e-8})
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x
            except Exception:
                pass
    
    # Now, compute the actual score for the best configuration found
    # The score is Delta / n^n where Delta is the product of distances after scaling to diameter 2
    pts = best_x.reshape(n, 2)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    mask = ~np.eye(n, dtype=bool)
    dists_masked = dists[mask]
    d_max = np.max(dists_masked)
    
    if d_max < 1e-15:
        # Fallback to regular octagon
        angles = 2 * np.pi * np.arange(n) / n
        return [[float(np.cos(a)), float(np.sin(a))] for a in angles]
    
    # Scale to diameter 2
    scale = 2.0 / d_max
    pts_scaled = pts * scale
    
    # Compute product of distances over ordered pairs
    diffs_scaled = pts_scaled[:, np.newaxis, :] - pts_scaled[np.newaxis, :, :]
    dists_scaled = np.sqrt(np.sum(diffs_scaled**2, axis=2))
    mask = ~np.eye(n, dtype=bool)
    dists_scaled_masked = dists_scaled[mask]
    
    # Score = prod(dists_scaled_masked) / n^n
    # Use log to avoid overflow/underflow
    log_score = np.sum(np.log(np.maximum(dists_scaled_masked, 1e-300))) - n * np.log(n)
    score = np.exp(log_score)
    
    # Ensure we return valid points
    result = [[float(pts_scaled[i, 0]), float(pts_scaled[i, 1])] for i in range(n)]
    
    # Check if score is reasonable, otherwise fallback
    if not np.isfinite(score) or score < 0.999:
        angles = 2 * np.pi * np.arange(n) / n
        return [[float(np.cos(a)), float(np.sin(a))] for a in angles]
    
    return result
# EVOLVE-BLOCK-END
