import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 250.0  # Safety margin
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        
        # Multi-start CMA-ES
        # Split budget into 5 runs of 100 evaluations each
        num_runs = 5
        budget_per_run = self.budget // num_runs
        
        for run_idx in range(num_runs):
            if time.time() > deadline:
                break
                
            # Initialize CMA-ES parameters
            d = self.dim
            lambda_ = 10
            mu = lambda_ // 2
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights = weights / np.sum(weights)
            
            # mu_eff
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Learning rates
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            c_c = (4 + mu_eff / lambda_) / (d + 4 + 2 * mu_eff / lambda_)
            c_1 = 2 / ((d + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((d + 2)**2 + mu_eff))
            
            # Initialization
            rng = np.random.default_rng(42 + run_idx)
            m = rng.uniform(-5, 5, d)
            sigma = 2.0
            C = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            
            # Precompute C^(1/2)
            eigvals, eigvecs = np.linalg.eigh(C)
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals))
            
            # CMA-ES Loop
            for gen in range(budget_per_run // lambda_):
                if time.time() > deadline:
                    break
                    
                # Sample lambda_ individuals
                samples = np.array([m + sigma * (sqrt_C @ rng.standard_normal(d)) for _ in range(lambda_)])
                # Clip to box
                samples = np.clip(samples, -5, 5)
                
                # Evaluate
                fitnesses = np.array([func(s) for s in samples])
                
                # Update global best
                min_idx = np.argmin(fitnesses)
                if fitnesses[min_idx] < best_f:
                    best_f = fitnesses[min_idx]
                    best_x = samples[min_idx]
                    
                # Sort indices by fitness
                sorted_indices = np.argsort(fitnesses)
                
                # Update mean
                m_new = np.sum(weights[:, np.newaxis] * samples[sorted_indices[:mu]], axis=0)
                
                # Update covariance matrix
                # y = (m_new - m) / sigma
                y = (m_new - m) / sigma
                
                # Cumulation step for sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * sqrt_C @ y
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / np.sqrt(d) - 1)) # Simplified, usually h_sigma check is needed
                
                # Cumulation step for C
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y
                h_pc = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c)**(2 * (gen + 1)))) < (1.4 + 2 / (d + 1)) * np.sqrt(d + 1)
                
                # Update C
                rank_mu = np.outer(y, y)
                rank_one = np.outer(pc, pc)
                C = (1 + c_1 * h_pc * (1 - c_c) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                
                # Update mean
                m = m_new
                
                # Update sqrt_C
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive definite
                eigvals = np.maximum(eigvals, 1e-10)
                sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals))
                
        return best_x, best_f
