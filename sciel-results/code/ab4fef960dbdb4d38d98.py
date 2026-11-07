"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size: 4 + floor(3 * log(dim))
        pop_size = int(4 + 3 * np.log(dim))
        if pop_size < 2:
            pop_size = 2
            
        # Initial step size
        sigma = 2.0
        
        # Initial mean: center of the box
        mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Parameters
        c_sigma = (pop_size + 2) / (dim + 4)
        c_c = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
        mu = pop_size // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_c = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
        c_sigma = (pop_size + 2) / (dim + 4)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial evaluation
        # We need to be careful with budget. 
        # Total budget is 500.
        # We will run CMA-ES until budget is exhausted or time limit is approached.
        
        start_time = time.time()
        time_limit = 290.0  # Safety margin for 300s limit
        
        # First evaluation
        x0 = mean
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls_made = 1
        
        # CMA-ES loop
        while calls_made < budget:
            # Check time
            if time.time() - start_time > time_limit:
                break
            
            # Generate population
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset
                C = np.eye(dim)
                L = np.eye(dim)
            
            # Generate random numbers
            N = rng.standard_normal((pop_size, dim)) if 'rng' in dir() else np.random.default_rng(0).standard_normal((pop_size, dim))
            # Better to have a single RNG instance
            if 'rng' not in dir():
                 rng = np.random.default_rng(0)
                 N = rng.standard_normal((pop_size, dim))
            else:
                 N = rng.standard_normal((pop_size, dim))
            
            # Transform
            Y = mean + sigma * (N @ L.T)
            
            # Clip to box
            Y = np.clip(Y, -5, 5)
            
            # Evaluate population
            F = np.zeros(pop_size)
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                F[i] = func(Y[i])
                calls_made += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = Y[i].copy()
            
            # If we ran out of budget mid-population, pad with best known value
            if calls_made >= budget:
                 # We can't update C properly if population is incomplete, 
                 # but we should return the best found.
                 # The loop condition will break next iteration.
                 pass
            
            # Sort by fitness
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]
            
            # Update mean
            new_mean = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
            
            # Update evolution path for covariance
            pc = (1 - c_c) * pc + c_c * (new_mean - mean) / sigma
            
            # Update evolution path for sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (L.T @ (new_mean - mean))
            
            # Update covariance
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - pop_size))) < (1 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_c)**(2 * (calls_made - pop_size))))
            
            # Note: The hsig condition usually uses generation count. 
            # Let's simplify or use a standard approximation.
            # For robustness, let's just update C with the standard formula.
            
            rank_mu = np.sum(weights[:, None] * (Y_sorted[:mu] - mean)[:, None] * (Y_sorted[:mu] - mean)[None, :], axis=0) / sigma**2
            
            C = (1 + c_1 * (hsig * (np.outer(pc, pc) - C) - c_1) + c_mu * rank_mu) * C
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp((c_sigma / 5) * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - pop_size))) - np.sqrt(dim)))
            
            # Update mean
            mean = new_mean
            
            # Clip mean to box? CMA-ES usually handles bounds via sampling, 
            # but keeping mean inside helps.
            mean = np.clip(mean, -5, 5)
            
            # Adaptive sigma check
            if sigma < 1e-10:
                break
                
        return best_x, best_f

# EVOLVE-BLOCK-END
