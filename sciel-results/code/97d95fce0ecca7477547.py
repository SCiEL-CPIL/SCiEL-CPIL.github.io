import numpy as np
import time
from collections import deque

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es_step(self, func, x_mean, C, sigma, lambda_, rng, calls, best_f, best_x):
        """
        Single CMA-ES iteration. Returns updated (x_mean, C, sigma, calls, best_f, best_x).
        """
        dim = len(x_mean)
        # Cholesky decomposition for sampling
        L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
        
        # Generate lambda_ offspring
        z = rng.standard_normal((lambda_, dim))
        offspring = x_mean + sigma * (z @ L.T)
        offspring = np.clip(offspring, -5.0, 5.0)
        
        # Evaluate offspring
        f_offspring = np.array([func(x) for x in offspring])
        calls += lambda_
        
        # Sort by fitness
        order = np.argsort(f_offspring)
        mu = max(1, lambda_ // 2)
        selected = order[:mu]
        
        # Weighted recombination
        weights = np.ones(mu) / mu
        weights /= weights.sum()
        
        # Update mean
        new_mean = np.sum(weights[:, None] * offspring[selected], axis=0)
        
        # Update covariance
        y_vecs = (offspring[selected] - x_mean) / sigma
        C = (1 - 2 * weights.sum() ** 2) * C + np.sum(weights[:, None, None] * y_vecs[:, :, None] * y_vecs[:, None, :], axis=0)
        
        # Step size adaptation (simple version)
        improvement = f_offspring[order[0]] - f_offspring[order[-1]]
        if improvement > 1e-10:
            # Converging: shrink sigma
            sigma *= 0.8
        else:
            # Plateau: grow sigma
            sigma *= 1.5
        
        # Clip sigma
        sigma = np.clip(sigma, 1e-8, 20.0)
        
        # Update best
        best_idx = np.argmin(f_offspring)
        if f_offspring[best_idx] < best_f:
            best_f = f_offspring[best_idx]
            best_x = offspring[best_idx].copy()
        
        return new_mean, C, sigma, calls, best_f, best_x

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Multi-start CMA-ES portfolio
        n_starts = 5
        lambda_ = 4  # offspring per generation
        calls = 0
        best_f = np.inf
        best_x = None
        
        # Initial random samples to get a good starting point
        for i in range(3):
            x0 = rng.uniform(-4.0, 4.0, dim)
            f0 = func(x0)
            calls += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1
        
        # Run multiple CMA-ES instances from different starting points
        for start_i in range(n_starts):
            if calls >= budget or time.time() > deadline:
                break
            
            # Initialize CMA-ES
            if start_i == 0:
                x_mean = best_x.copy()
            else:
                x_mean = rng.uniform(-4.0, 4.0, dim)
                fx_mean = func(x_mean)
                calls += 1
                if fx_mean < best_f:
                    best_f = fx_mean
                    best_x = x_mean.copy()
            
            C = np.eye(dim)
            sigma = 2.0
            
            # Run CMA-ES for a portion of the budget
            remaining = budget - calls
            max_iters = min(remaining // lambda_, 150)
            
            for iter_i in range(max_iters):
                if calls >= budget or time.time() > deadline:
                    break
                
                x_mean, C, sigma, calls, best_f, best_x = self._cma_es_step(
                    func, x_mean, C, sigma, lambda_, rng, calls, best_f, best_x
                )
                
                # Restart if sigma is too small or too large
                if sigma < 1e-5 or sigma > 15.0:
                    break
        
        # Final local refinement around best found
        if calls < budget and best_x is not None:
            x_curr = best_x.copy()
            f_curr = best_f
            sigma = 0.1
            
            while calls < budget and time.time() < deadline:
                y = x_curr + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls += 1
                
                if fy < f_curr - 1e-12:
                    x_curr = y
                    f_curr = fy
                    sigma *= 0.7
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                else:
                    sigma *= 1.3
                    if sigma > 5.0:
                        break

        return best_x, best_f
