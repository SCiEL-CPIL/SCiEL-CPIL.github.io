"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    if points.shape[0] < 2:
        return 1e10
    dists = pdist(points)
    if len(dists) == 0:
        return 1e11
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d < 1e-10:
        return 1e11
    return (max_d / min_d) ** 2

def construct():
    n = N
    d = D
    
    # Precompute indices for pairwise distances to speed up objective if needed,
    # but pdist is usually fast enough for n=14.
    
    def objective_log_ratio(x):
        """
        Objective function: log(max_dist^2) - log(min_dist^2)
        Approximated using log-sum-exp for smoothness.
        """
        X = x.reshape(n, d)
        d_sq = pdist(X) ** 2
        
        # Ensure distances are not too small to avoid log(0) issues in the sum
        # d_sq can be 0 if points coincide.
        # We add a small epsilon to d_sq for the min calculation stability?
        # No, log-sum-exp for min is -T log sum exp(-d/T). If d=0, exp(0)=1.
        
        # Temperatures
        # Scale of distances is arbitrary. We can normalize X in the objective?
        # No, the ratio is scale invariant. But the optimizer needs a gradient.
        # The gradient of the ratio is zero if we scale all points?
        # d_i -> c d_i. Ratio is invariant.
        # So the objective is flat along the radial direction.
        # We should fix the scale to help the optimizer.
        # e.g., sum of squared norms = 1.
        # But that adds a constraint.
        # Alternatively, just let the optimizer find a scale.
        # The log-sum-exp terms will dominate.
        
        # Let's try to normalize the points inside the objective to a fixed scale
        # to provide a better landscape, but this breaks the exact ratio invariance
        # if we don't do it carefully.
        # Actually, since the ratio is scale invariant, any scale is fine.
        # But the log-sum-exp approximation depends on the scale relative to T.
        # Let's compute the mean squared distance to set T.
        
        if np.all(d_sq < 1e-12):
            return 1e10
            
        # Dynamic temperature based on current scale
        # T should be small relative to the spread of distances.
        # Let's use a fixed T for simplicity first, maybe 0.1 or 0.05.
        # If the points are very close, d_sq is small, exp(d/T) ~ 1 + d/T.
        # If points are far, d_sq is large.
        
        # Better: Normalize the points so that the sum of squared distances is 1.
        # This makes the problem well-scaled.
        sum_d_sq = np.sum(d_sq)
        if sum_d_sq < 1e-12:
             return 1e10
             
        # We don't actually need to normalize X, we can just adjust T.
        # Or we can normalize X inside the objective calculation for the gradient?
        # No, x is the variable.
        
        # Let's stick to a fixed T but choose it reasonably.
        # The optimal configuration for 14 points in 3D will have distances of order 1.
        T = 0.05
        
        # Max approximation: T * log(sum(exp(d_sq / T)))
        # Min approximation: -T * log(sum(exp(-d_sq / T)))
        
        # For numerical stability, subtract max/min from exponents
        d_sq_max = np.max(d_sq)
        d_sq_min = np.min(d_sq)
        
        # Log-sum-exp for max
        # log(sum(exp(d/T))) = log(sum(exp((d - d_max)/T))) + d_max/T
        shifted_max = d_sq - d_sq_max
        log_sum_exp_max = d_sq_max / T + np.log(np.sum(np.exp(shifted_max / T)))
        
        # Log-sum-exp for min
        # log(sum(exp(-d/T))) = log(sum(exp(-(d - d_min)/T))) - d_min/T
        shifted_min = d_sq - d_sq_min
        log_sum_exp_min = -d_sq_min / T + np.log(np.sum(np.exp(-shifted_min / T)))
        
        # Approximation of log(max) and log(min)
        # log(max_d) = 0.5 * log(max_d_sq)
        # log(min_d) = 0.5 * log(min_d_sq)
        # Ratio squared = max_d_sq / min_d_sq
        # log(Ratio^2) = log(max_d_sq) - log(min_d_sq)
        
        log_max_d_sq = T * log_sum_exp_max
        log_min_d_sq = T * log_sum_exp_min
        
        if log_min_d_sq < 1e-12:
            return 1e10
            
        return log_max_d_sq - log_min_d_sq

    def objective_ratio(x):
        """
        Direct ratio objective, less smooth but more direct.
        """
        X = x.reshape(n, d)
        d_sq = pdist(X) ** 2
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        if min_d_sq < 1e-10:
            return 1e10
        return max_d_sq / min_d_sq

    best_score = 1e10
    best_points = None
    
    # Multiple starts
    rng = np.random.RandomState(42)
    
    starts = []
    
    # 1. Random Gaussian
    starts.append(rng.randn(n, d))
    
    # 2. Random points on sphere
    X_sphere = rng.randn(n, d)
    norms = np.linalg.norm(X_sphere, axis=1, keepdims=True)
    starts.append(X_sphere / norms)
    
    # 3. Simplex-like (first n vertices of high-dim simplex projected?)
    # Just use random points with larger variance
    starts.append(rng.randn(n, d) * 5)
    
    # 4. Grid-like initialization
    # Spread points in a grid
    coords = []
    # 14 points. 2x2x3 grid + 2 extras?
    # Let's just use a few random variations
    for _ in range(5):
         starts.append(rng.randn(n, d) * 2)

    for x0 in starts:
        try:
            # Use L-BFGS-B for the log-ratio objective (smoother)
            res = minimize(objective_log_ratio, x0.flatten(), method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            x_opt = res.x.reshape(n, d)
            score = _compute_score(x_opt)
            
            if score < best_score:
                best_score = score
                best_points = x_opt
                
            # Refine with direct ratio objective if we got close
            if score < 3.0: # Threshold for refinement
                 try:
                     res2 = minimize(objective_ratio, x_opt.flatten(), method='BFGS',
                                     options={'maxiter': 200, 'gtol': 1e-8})
                     x_opt2 = res2.x.reshape(n, d)
                     score2 = _compute_score(x_opt2)
                     if score2 < best_score:
                         best_score = score2
                         best_points = x_opt2
                 except Exception:
                     pass
                     
        except Exception as e:
            pass

    if best_points is None:
        # Fallback
        return rng.randn(n, d)

    return best_points

# EVOLVE-BLOCK-END
