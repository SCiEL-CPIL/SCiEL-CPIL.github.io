"""Rewritten optimizer using CMA-ES style covariance adaptation and population-based search."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Population-based continuous optimizer with adaptive step sizes (CMA-ES inspired)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)

        # Initial population size
        lambda_pop = max(4, dim + 3)
        
        # Initial mean: start at center of box
        mean = np.zeros(dim)
        
        # Initial step size
        sigma = 3.0  # Start large to explore [-5, 5]
        
        # Track best solution found
        best_x = mean.copy()
        best_f = np.inf
        
        # Evaluate initial mean
        f_mean = func(mean)
        if f_mean < best_f:
            best_f = f_mean
            best_x = mean.copy()
        
        # Evolution path for sigma adaptation
        ps = np.zeros(dim)
        cs = 1.0 / (5 + dim / 2)
        
        # Covariance matrix
        C = np.eye(dim)
        # Eigen decomposition cache
        D = np.ones(dim)
        B = np.eye(dim)
        
        # Update counters
        gen = 0
        
        # Learning rates
        cmu = 2 / ((dim + 4)**2 + 10)
        c_sigma = 2 / ((dim + 1.3)**2 + 5)
        
        # Number of function calls used so far
        calls_used = 1  # for the initial mean evaluation
        
        # Main loop
        while calls_used < budget:
            # Determine how many offspring we can afford
            # We need at least 1 call per offspring
            remaining = budget - calls_used
            
            # Population size for this generation
            pop_size = min(lambda_pop, remaining)
            
            if pop_size < 1:
                break
            
            # Update step size based on evolution path
            # Simplified CMA-ES sigma update
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * lambda_pop) * B @ (D * mean / sigma)
            # Note: The above is a simplification. Let's use a simpler adaptive strategy.
            
            # Simpler adaptive sigma:
            # If we found improvements in the last generation, increase sigma slightly
            # Otherwise decrease it.
            
            # Generate offspring
            offspring = []
            offspring_f = []
            
            for i in range(pop_size):
                # Sample from multivariate normal
                # For simplicity and speed, use diagonal covariance initially
                # z ~ N(0, I)
                z = rng.standard_normal(dim)
                x_new = mean + sigma * z
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                calls_used += 1
                
                offspring.append(x_new)
                offspring_f.append(f_new)
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                
                # If we run out of budget, break
                if calls_used >= budget:
                    break
            
            if len(offspring) == 0:
                break
            
            offspring = np.array(offspring)
            offspring_f = np.array(offspring_f)
            
            # Sort offspring by fitness
            idx = np.argsort(offspring_f)
            sorted_offspring = offspring[idx]
            sorted_f = offspring_f[idx]
            
            # Select top mu individuals
            mu = max(1, len(sorted_offspring) // 2)
            selected = sorted_offspring[:mu]
            
            # Compute new mean
            # Weights: linear weights
            w = np.arange(1, mu + 1)
            w = w / w.sum()
            
            new_mean = np.sum(w[:, None] * selected, axis=0)
            
            # Update covariance matrix (simplified)
            # In full CMA-ES, we update C based on the selected points
            # Here, we use a simple rank-mu update approximation
            # For efficiency, let's just track the variance of the selected points
            if mu > 1:
                # Estimate covariance from selected points
                # Center the selected points
                centered = selected - new_mean
                # Covariance estimate
                C_new = (centered.T @ (w[:, None] * centered)) / (sigma ** 2)
                # Add regularization to keep it positive definite
                C = 0.9 * C + 0.1 * (C_new + np.eye(dim) * 1e-6)
            else:
                C = C  # No update if only one selected
            
            # Update step size
            # If the best offspring is much better than the mean, increase sigma
            # Otherwise, decrease sigma
            if sorted_f[0] < f_mean - 1e-8:
                sigma *= 1.1  # Slight increase if improving
            else:
                sigma *= 0.95  # Slight decrease if not improving
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 0.01, 10.0)
            
            # Update mean and fitness
            mean = new_mean
            f_mean = func(mean)
            calls_used += 1
            
            if f_mean < best_f:
                best_f = f_mean
                best_x = mean.copy()
            
            # If we've used all budget, break
            if calls_used >= budget:
                break
            
            gen += 1
        
        return best_x, best_f
# EVOLVE-BLOCK-END
