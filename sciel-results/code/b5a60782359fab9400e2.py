# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min / d_max)^2.
    
    Returns
        points: np.ndarray of shape (14,3)
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep safety margin
    
    n = 14
    d = 3
    
    # Scale invariant problem: we can fix one point at origin, or use sphere
    # Actually, the ratio is scale-invariant. Let's work on the unit sphere?
    # No, the points don't have to be on a sphere.
    # However, we can normalize: translate centroid to 0, scale so max dist is 1?
    # Or just optimize directly.
    
    # A good strategy:
    # 1. Use a smooth surrogate for the min/max objective.
    #    We want to maximize min(d_ij) / max(d_ij).
    #    Let f(x) = log(min(d_ij)) - log(max(d_ij)).
    #    Or maximize (min/max)^2.
    #    
    #    Soft-min: -beta * log(sum(exp(-beta * d_ij^2))) approximates min(d_ij^2)
    #    Soft-max: beta * log(sum(exp(beta * d_ij^2))) approximates max(d_ij^2)
    #    
    #    Objective to maximize: (soft_min / soft_max)^2
    #    Or minimize: -2 * (log(soft_min) - log(soft_max))
    
    def objective(x, beta):
        """
        x: flattened array of shape (14*3,)
        beta: parameter for soft min/max
        Returns negative of the objective (since we minimize)
        """
        points = x.reshape(n, d)
        # Compute pairwise squared distances
        # Using pdist is convenient but returns 1D array
        d2 = pdist(points, 'sqeuclidean')
        
        # Soft min of d2
        # min(d2) approx -1/beta * log(sum(exp(-beta * d2)))
        # To avoid overflow, subtract max
        d2_max = np.max(d2)
        d2_min = np.min(d2)
        
        # If all points are coincident, d2_max = 0
        if d2_max < 1e-10:
            return 0.0  # Metric is 0, so -metric is 0? Wait, we want to maximize metric.
            # If metric is 0, we want to minimize a large number?
            # Let's return -log(metric). If metric=0, log is -inf.
            # Better: return - (log(soft_min) - log(soft_max))
            # If soft_min ~ 0, log is -inf, so -(-inf - ...) = inf. Good.
        
        # Soft min
        # exp(-beta * (d2 - d2_max)) to prevent overflow?
        # Actually, standard log-sum-exp:
        # log(sum(exp(-beta * d2))) = -beta * d2_max + log(sum(exp(-beta * (d2 - d2_max))))
        # soft_min = -1/beta * log(sum(exp(-beta * d2)))
        #          = d2_max - 1/beta * log(sum(exp(-beta * (d2 - d2_max))))
        
        # Soft max
        # log(sum(exp(beta * d2))) = beta * d2_max + log(sum(exp(beta * (d2 - d2_max))))
        # soft_max = 1/beta * log(sum(exp(beta * d2)))
        #          = d2_max + 1/beta * log(sum(exp(beta * (d2 - d2_max))))
        
        # Let's compute log_soft_min and log_soft_max
        # log(soft_min) = log(d2_max - 1/beta * log(S_min))
        # This is getting complicated.
        
        # Alternative: directly maximize (min/max)^2.
        # Let's use a simpler smooth approximation.
        # f(x) = (sum(exp(-beta*d2)) / sum(exp(beta*d2)))^(-1/beta * 2) ? No.
        
        # Let's stick to: maximize log(min(d)) - log(max(d))
        # Approximate log(min(d)) with -1/beta * log(sum(exp(-beta*d)))
        # Approximate log(max(d)) with 1/beta * log(sum(exp(beta*d)))
        
        # But we have d, not d2. pdist gives d or d2.
        # Let's use d2 to avoid sqrt.
        # log(d) = 0.5 * log(d2).
        # So we want to maximize 0.5 * (log(min(d2)) - log(max(d2)))
        
        # Soft min of d2:
        # m = -1/beta * log(sum(exp(-beta * d2)))
        # Soft max of d2:
        # M = 1/beta * log(sum(exp(beta * d2)))
        
        # We want to maximize 0.5 * (log(m) - log(M))
        # Minimize -0.5 * (log(m) - log(M))
        
        # Compute m and M safely
        # For m:
        max_d2 = np.max(d2)
        # exp(-beta * (d2 - max_d2)) is safe (<= 1)
        s_min = np.sum(np.exp(-beta * (d2 - max_d2)))
        # m = max_d2 - (1/beta) * log(s_min)
        # log(m) = log(max_d2 - (1/beta) * log(s_min))
        
        # For M:
        # exp(beta * (d2 - max_d2)) is safe (<= 1)
        s_max = np.sum(np.exp(beta * (d2 - max_d2)))
        # M = max_d2 + (1/beta) * log(s_max)
        # log(M) = log(max_d2 + (1/beta) * log(s_max))
        
        if max_d2 < 1e-10:
            return 1e10  # Penalize coincident points heavily
        
        # m is approx min(d2), M is approx max(d2)
        # Note: m can be negative if beta is small and s_min is large?
        # s_min >= 1 (since one term is exp(0)=1). log(s_min) >= 0.
        # m = max_d2 - (1/beta)*log(s_min).
        # If max_d2 is small and beta is small, m could be negative?
        # But min(d2) >= 0.
        # Let's just compute m and M and check.
        
        m = max_d2 - (1.0/beta) * np.log(s_min)
        M = max_d2 + (1.0/beta) * np.log(s_max)
        
        if m <= 0 or M <= 0:
            return 1e10
        
        # Objective to minimize: -0.5 * (log(m) - log(M))
        val = -0.5 * (np.log(m) - np.log(M))
        return val
    
    # Initial guesses
    best_points = None
    best_score = -np.inf
    
    # 1. Random starts
    np.random.seed(42)
    for i in range(20):
        if time.time() > deadline - 60:
            break
        x0 = np.random.randn(n * d)
        # Normalize?
        # The problem is scale invariant.
        # Let's project to unit sphere? No, points can be anywhere.
        # But scaling doesn't change the ratio.
        # Let's just run the optimizer.
        
        # Increase beta gradually to sharpen the min/max
        betas = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        current_x = x0.copy()
        
        for beta in betas:
            if time.time() > deadline - 60:
                break
            
            # Use Nelder-Mead or SLSQP?
            # SLSQP is better for smooth objectives.
            # But we need gradients for SLSQP.
            # Let's use Nelder-Mead first, then SLSQP with numerical gradient?
            # Or just Nelder-Mead with tight tolerances.
            
            res = minimize(objective, current_x, args=(beta,), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            current_x = res.x
            
            # Evaluate true score
            pts = current_x.reshape(n, d)
            d = pdist(pts)
            if len(d) == 0 or np.any(~np.isfinite(d)):
                continue
            d_min = np.min(d)
            d_max = np.max(d)
            if d_max == 0:
                score = 0
            else:
                score = (d_min / d_max) ** 2
            
            if score > best_score:
                best_score = score
                best_points = pts.copy()
    
    # 2. Try structured starts
    # e.g., vertices of a regular polytope?
    # For 14 points in 3D, maybe a subset of icosahedron + others?
    # Or just random is fine.
    
    # Final polish with high beta
    if best_points is not None:
        x0 = best_points.flatten()
        for beta in [100.0, 200.0, 500.0]:
            if time.time() > deadline - 30:
                break
            res = minimize(objective, x0, args=(beta,), method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
            current_x = res.x
            pts = current_x.reshape(n, d)
            d = pdist(pts)
            if len(d) == 0 or np.any(~np.isfinite(d)):
                continue
            d_min = np.min(d)
            d_max = np.max(d)
            if d_max == 0:
                score = 0
            else:
                score = (d_min / d_max) ** 2
            
            if score > best_score:
                best_score = score
                best_points = pts.copy()
    
    if best_points is None:
        # Fallback: random
        np.random.seed(42)
        best_points = np.random.randn(n, d)
    
    # Ensure finite
    best_points = np.nan_to_num(best_points, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return best_points

# EVOLVE-BLOCK-END
