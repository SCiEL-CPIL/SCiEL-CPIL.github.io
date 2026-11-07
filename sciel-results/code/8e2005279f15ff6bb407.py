import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 250.0  # Safety margin
        
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 6  # Population size
        mu = lam // 2  # Number of best individuals used for update
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        
        # mu_eff
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / lam) / (d + 4 + 2 * mu_eff / lam)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Initialization
        rng = np.random.default_rng(42)
        m = rng.uniform(-5, 5, d) # Start random within box
        sigma = 2.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Precompute C^(1/2)
        eigvals, eigvecs = np.linalg.eigh(C)
        sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals))
        
        best_x = m
        best_f = func(m)
        budget_used = 1
        
        # CMA-ES Loop
        while budget_used < budget:
            if time.time() > deadline:
                break
                
            # Determine how many individuals to sample in this generation
            # We can sample lam individuals each generation
            # If remaining budget < lam, we might just sample the rest
            # But CMA-ES needs a full population for update.
            # So we run full generations until budget is exhausted.
            
            # Sample lambda_ individuals
            samples = np.array([m + sigma * (sqrt_C @ rng.standard_normal(d)) for _ in range(lam)])
            # Clip to box
            samples = np.clip(samples, -5, 5)
            
            # Check if we have enough budget for a full generation
            if budget_used + lam > budget:
                # We can't afford a full generation.
                # We could sample the remaining budget individually, but CMA-ES update is not valid with partial population.
                # Strategy: If we can't do a full generation, we just do random search or stop.
                # Given 500 is divisible by 6? 500/6 = 83.33.
                # So we will have 83 full generations (498 evaluations) and 2 left.
                # Let's just run full generations.
                break
            
            # Evaluate
            fitnesses = np.array([func(s) for s in samples])
            budget_used += lam
            
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
            y = (m_new - m) / sigma
            
            # Cumulation step for sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * sqrt_C @ y
            # Check for sigma update (simplified, usually h_sigma check)
            # For simplicity and robustness, we often skip the h_sigma check in simple implementations,
            # or use a basic norm check.
            # Let's use the standard check:
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (budget_used // lam + 1)))) < (1.4 + 2 / (d + 1)) * np.sqrt(d + 1)
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / np.sqrt(d) - 1) * h_sigma)
            
            # Cumulation step for C
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            h_pc = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c)**(2 * (budget_used // lam + 1)))) < (1.4 + 2 / (d + 1)) * np.sqrt(d + 1)
            
            # Update C
            rank_mu = np.outer(y, y)
            rank_one = np.outer(pc, pc)
            C = (1 + c_1 * h_pc * (1 - c_c) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Update mean
            m = m_new
            
            # Update sqrt_C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals))
            
        return best_x, best_f
