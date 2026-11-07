"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """Bayesian Optimization with Nearest Neighbor Kriging surrogate."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 280.0  # Safety margin

    def __call__(self, func):
        rng = np.random.default_rng(42)
        x = np.zeros(self.dim)
        fx = float('inf')
        
        # Storage for observations
        X_obs = []
        F_obs = []
        
        # Parameters
        K_neighbors = 10
        sigma_length = 1.5  # Kernel length scale
        sigma_noise = 1e-6  # Small noise for stability
        sigma_func = 1.0    # Scale for variance estimation
        
        def predict(x_pred):
            """Predict mean and variance using K nearest neighbors."""
            if len(X_obs) == 0:
                return 0.0, 1.0 # Prior: mean 0, var 1 (arbitrary)
            
            X_arr = np.array(X_obs)
            F_arr = np.array(F_obs)
            
            # Compute distances
            dists = np.linalg.norm(X_arr - x_pred, axis=1)
            
            # Select K nearest
            if len(X_obs) > K_neighbors:
                idx = np.argpartition(dists, K_neighbors)[:K_neighbors]
            else:
                idx = np.arange(len(X_obs))
                
            d_k = dists[idx]
            f_k = F_arr[idx]
            
            # Kernel weights
            # Use RBF kernel
            w = np.exp(-d_k**2 / (2 * sigma_length**2))
            
            # Handle case where weights are very small
            w_sum = np.sum(w)
            if w_sum < 1e-10:
                w_sum = 1e-10
                w = np.ones_like(w)
            
            # Weighted mean
            mu = np.sum(w * f_k) / w_sum
            
            # Weighted variance
            # Variance of the weighted distribution
            var_f = np.sum(w * (f_k - mu)**2) / w_sum
            
            # Uncertainty from distance: closer points -> lower variance
            # If we are far from all points, variance should be high
            min_dist = np.min(d_k)
            # Scale uncertainty by how "covered" we are
            # 1 - max_weight is a measure of how far we are from the best neighbor
            coverage = np.max(w)
            var = sigma_func**2 * (1 - coverage) + sigma_noise**2
            
            # Combine data variance and distance variance?
            # Let's just use distance-based variance primarily, as data variance is local
            # A simple heuristic: var = sigma_func**2 * (1 - coverage)
            # This ensures var goes to 0 at observed points (coverage=1)
            
            return mu, var

        def expected_improvement(x_eval, mu, var, f_best):
            """Compute Expected Improvement for minimization."""
            if var < 1e-10:
                return 0.0
            s = np.sqrt(var)
            z = (f_best - mu) / s
            ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
            return ei

        def neg_ei(x_eval):
            """Negative EI for minimization."""
            mu, var = predict(x_eval)
            ei = expected_improvement(x_eval, mu, var, fx)
            return -ei

        # 1. Initialization
        n_init = 15
        for _ in range(n_init):
            if time.time() > self.deadline:
                break
            x_try = rng.uniform(-5, 5, self.dim)
            f_try = func(x_try)
            X_obs.append(x_try)
            F_obs.append(f_try)
            if f_try < fx:
                fx = f_try
                x = x_try.copy()
                
        # 2. Main Loop
        for i in range(self.budget - len(X_obs)):
            if time.time() > self.deadline:
                break
                
            # Find next point by maximizing EI
            # Use multiple restarts of Nelder-Mead on -EI
            best_x_next = None
            best_ei_val = -np.inf
            
            # Start points: current best, random points
            start_points = [x.copy()]
            for _ in range(5):
                start_points.append(rng.uniform(-5, 5, self.dim))
            
            for sp in start_points:
                # Clip start point to bounds
                sp = np.clip(sp, -5, 5)
                res = minimize(neg_ei, sp, method='Nelder-Mead', 
                               options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 100})
                
                if res.success or res.fun < best_ei_val:
                    # Check if the result is valid
                    if np.all(res.x >= -5) and np.all(res.x <= 5):
                        if res.fun < best_ei_val:
                            best_ei_val = res.fun
                            best_x_next = res.x.copy()
            
            if best_x_next is None:
                # Fallback: random
                best_x_next = rng.uniform(-5, 5, self.dim)
            else:
                best_x_next = np.clip(best_x_next, -5, 5)
                
            # Evaluate
            f_next = func(best_x_next)
            X_obs.append(best_x_next)
            F_obs.append(f_next)
            
            if f_next < fx:
                fx = f_next
                x = best_x_next.copy()
                
        return x, fx
# EVOLVE-BLOCK-END
