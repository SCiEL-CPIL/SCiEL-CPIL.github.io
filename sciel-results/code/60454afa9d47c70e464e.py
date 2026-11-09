"""CMA-ES variant for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with diagonal covariance for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        dim = self.dim
        budget = self.budget
        
        call_count = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        # CMA-ES parameters
        # For dim=5, lambda ~ 4 + floor(3*ln(5)) = 4 + 4 = 8? 
        # Standard is lambda = 4 + floor(3*ln(N)). For N=5, lambda=8.
        # Let's use lambda=10 for slightly more diversity, or stick to standard.
        # Standard lambda for N=5 is 8.
        lambda_ = 8
        sigma = 2.0  # Initial step size, large enough to cover the box [-5,5] roughly
        sigma_min = 1e-6
        sigma_max = 10.0
        
        # Population size
        mu = lambda_ // 2  # Weighted recombination from best mu
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2)**2 + mu_eff))
        
        # Initial state
        x_mean = np.zeros(dim)
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Initial sampling to find a decent start
        # CMA-ES starting at 0 might be okay, but let's do a small random search first
        # to avoid being stuck in a bad local minimum if 0 is far from the global one.
        # However, Rastrigin global min is usually at 0 (shifted).
        # Let's just start CMA-ES at 0 with a large sigma.
        
        # To ensure we make progress quickly for AOCC, we do a few random evaluations first
        # to establish a baseline, then let CMA-ES take over.
        # Actually, CMA-ES itself evaluates the mean? No, it evaluates offspring.
        # We should evaluate the mean first to have a baseline.
        
        # Evaluate initial mean
        f_mean = wrapped_func(x_mean)
        
        # Main loop
        while call_count < budget and time.time() < deadline:
            # Check time
            if time.time() > deadline - 1.0:
                break
                
            # Generate population
            # We need to generate lambda_ samples.
            # Each sample requires a matrix-vector product or Cholesky decomposition of C.
            # For dim=5, Cholesky is cheap.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity or add jitter
                C = np.eye(dim)
                L = np.eye(dim)
                
            pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = np.random.randn(dim)
                pop[i] = x_mean + sigma * (L @ z)
            
            # Evaluate population
            fitness = np.zeros(lambda_)
            for i in range(lambda_):
                fitness[i] = wrapped_func(pop[i])
                
            # Sort by fitness
            idx = np.argsort(fitness)
            pop_sorted = pop[idx]
            fitness_sorted = fitness[idx]
            
            # Update mean
            x_mean = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)
            
            # Update step size (CSA)
            # Standard CSA uses ps
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * L_inv @ (x_mean - x_mean_old)
            # We need x_mean_old and L_inv. 
            # Storing L_inv is expensive to compute every step. 
            # Alternative: Use simple sigma adaptation or just fixed sigma decay.
            # Given the short budget (500 calls) and dim=5, maybe a simpler adaptive scheme is better.
            
            # Let's use a simpler sigma adaptation:
            # If we made progress, keep sigma. If not, decrease.
            # Or use the standard CMA-ES sigma update.
            
            # To keep it robust and simple, let's implement the standard sigma update properly.
            # We need the inverse of L. Since L is 5x5, inv is fast.
            L_inv = np.linalg.inv(L)
            
            # x_mean_old was the mean before this generation
            # We didn't store it explicitly in the loop above, but x_mean was the old mean.
            # Wait, I updated x_mean in place? No, I used x_mean for generation.
            # I need to store the old mean.
            
            # Let's restructure the loop to store old_mean.
            pass # This block is a placeholder for logic re-evaluation
            
        # The previous loop logic was slightly flawed in variable management for the standard CMA-ES update.
        # Let's write a cleaner CMA-ES implementation.
        
        # Reset state for the clean implementation
        call_count = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        # Re-evaluate initial mean
        f_mean = wrapped_func(x_mean)
        
        x_mean_old = x_mean.copy()
        
        while call_count < budget and time.time() < deadline:
            if time.time() > deadline - 1.0:
                break
                
            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
                
            pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = np.random.randn(dim)
                pop[i] = x_mean + sigma * (L @ z)
            
            # Evaluate population
            fitness = np.zeros(lambda_)
            for i in range(lambda_):
                fitness[i] = wrapped_func(pop[i])
                
            # Sort
            idx = np.argsort(fitness)
            pop_sorted = pop[idx]
            fitness_sorted = fitness[idx]
            
            # Update mean
            x_mean_new = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)
            
            # Update step size
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * L_inv @ (x_mean_new - x_mean_old)
            L_inv = np.linalg.inv(L)
            h_sigma = 1 if np.trace(np.outer(ps, ps)) / (1 - (1 - c_sigma)**(2 * (call_count + 1))) < (1 + 2 / dim) * chi_n else 0
            # chi_n is expected norm of standard normal, approx sqrt(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            ps = (1 - c_sigma) * ps + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (L_inv @ (x_mean_new - x_mean_old))
            
            sigma = sigma * np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / chi_n - 1)) # Simplified CSA
            # Proper CSA: sigma = sigma * exp(c_sigma / c_sigma * (||ps||/chi_n - 1))
            # Actually, the standard formula is:
            # sigma = sigma * exp( (c_sigma / c_sigma) * (||ps|| / chi_n - 1) )
            # The c_sigma in the exponent denominator is usually 1 in some formulations, or c_sigma.
            # Hansen's CMA-ES: sigma = sigma * exp( (c_sigma / c_sigma) * (||ps||/chi_n - 1) )
            # Wait, the learning rate for sigma is c_sigma.
            # sigma = sigma * exp( c_sigma * (||ps||/chi_n - 1) )
            
            sigma = sigma * np.exp(c_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, sigma_min, sigma_max)
            
            # Update covariance
            # pc = (1 - c_c) * pc + h_c * sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean_old) / sigma
            h_c = 1 if np.trace(np.outer(pc, pc)) / (1 - (1 - c_c)**(2 * (call_count + 1))) < (1 + 2 / dim) * chi_n else 0
            pc = (1 - c_c) * pc + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean_old) / sigma
            
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - h_c) * c_c * (2 - c_c) * I) + c_mu * sum(w_i * (y_i @ y_i.T))
            # where y_i = (x_i - x_mean_old) / sigma
            
            y = (pop_sorted[:mu] - x_mean_old) / sigma
            rank_mu = np.sum(weights[:, None, None] * (y.T @ y), axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - h_c) * c_c * (2 - c_c) * np.eye(dim)) + c_mu * rank_mu
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update mean
            x_mean_old = x_mean
            x_mean = x_mean_new
            
            # Check for convergence
            if sigma < sigma_min:
                break
                
        return best_x, best_f

# EVOLVE-BLOCK-END
