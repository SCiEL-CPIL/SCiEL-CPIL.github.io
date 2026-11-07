import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = 330.0
        
        N = self.dim
        # Standard popsize for CMA-ES
        popsize = 4 + int(3 * np.log(N))
        
        # Initialize
        x = np.random.uniform(-5, 5, N)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1
        
        sigma = 2.0
        C = np.eye(N)
        pc = np.zeros(N)
        
        # CMA-ES parameters
        c_sigma = 1.0 / (N + 3)
        c_c = (4 - N + 1) / (N + 4 + 3)
        d_sigma = 1 + 2 * max(0, np.sqrt((N + 1) / (N + 3)) - 1) + c_sigma
        chi_N = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N * N))
        
        # Weights
        w = np.log(popsize / 2 + 0.5) - np.log(np.arange(1, popsize + 1))
        w /= w.sum()
        mu_eff = 1.0 / (w @ w)
        
        # Update rates
        c_cov = (2 + 2 * min(1, (mu_eff - 2) / (N + 2))) / ((N + 1.3) ** 2 + mu_eff)
        
        for i in range(self.budget - 1):
            # Time guard
            if time.time() > deadline - 2.0:
                break
                
            # Check if we have enough budget for a full generation
            if calls + popsize > self.budget:
                break
                
            # Sample
            z = np.random.randn(popsize, N)
            L = np.linalg.cholesky(C)
            y = z @ L.T
            x_pop = x + sigma * y
            x_pop = np.clip(x_pop, -5, 5)
            
            # Evaluate
            f_pop = np.empty(popsize)
            for j in range(popsize):
                f_pop[j] = func(x_pop[j])
                calls += 1
                
            # Update best
            idx_min = np.argmin(f_pop)
            if f_pop[idx_min] < best_fx:
                best_fx = f_pop[idx_min]
                best_x = x_pop[idx_min].copy()
                
            # Sort
            order = np.argsort(f_pop)
            y_sorted = y[order]
            
            # Update mean
            # Only top half is used for mean update in standard CMA-ES
            x_new = x + sigma * (w[:popsize // 2] @ y_sorted[:popsize // 2])
            x_new = np.clip(x_new, -5, 5)
            
            # Update evolution paths
            # p_sigma
            pc_sigma = (1 - c_sigma) * pc_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x_new - x)
            # Wait, pc_sigma is not used in the simplified version above, I need to track it.
            # Let's use the standard update rules properly.
            
            # Recalculating paths properly
            # p_sigma update
            # We need to store p_sigma from previous iteration.
            # Let's re-initialize p_sigma = 0.
            
            # Actually, let's just implement the standard CMA-ES update block.
            
            # 1. Update mean
            # x_new is calculated above
            
            # 2. Update p_sigma
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x_new - x)
            # Note: The vector (x_new - x) / sigma is the weighted sum of y_sorted[:mu]
            # Let's call this v_sigma
            v_sigma = w[:popsize // 2] @ y_sorted[:popsize // 2]
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v_sigma
            
            # 3. Update p_c
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1 - c_sigma)^(2*(i_gen+1))) < (1.4 + 2/(N+1)) * chi_N else 0
            # This requires tracking i_gen.
            
            # This is getting complex to implement from scratch in a short block.
            # I will use a simplified but effective CMA-ES variant or ensure the implementation is correct.
            
            # Let's refine the implementation to be robust.
            
            # State variables to persist
            # p_sigma, p_c, C, sigma, x
            
            # Since I am rewriting, I will define them before the loop.
            
            pass # Placeholder, I will write the full code in the final block.
            
        return best_x, best_fx
