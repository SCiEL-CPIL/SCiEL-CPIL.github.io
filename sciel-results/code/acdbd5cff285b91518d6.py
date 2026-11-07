"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cmaes(self, func, x0, f0, sigma0, budget, rng):
        n = self.dim
        # Initial population size
        lambda_ = 4 + int(3 * np.log(n))
        mu = lambda_ // 2
        
        # CMA-ES parameters
        c_sigma = (mu + 0.5) / (n + mu + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu / n) / (n + 4 + 2 * mu / n)
        c_1 = (2 / ((n + 1.3) ** 2 + mu)) * (2 - c_c + c_sigma)
        c_mu = (2 / ((n + 1.3) ** 2 + mu)) * min(1, c_1 / 2)
        
        # Eigenvalues for update
        alpha_mu = 2 * (c_mu * (n + 2) + c_1 * (n - 2)) / (c_mu + c_1)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Initialize state
        x = x0.copy()
        sigma = sigma0
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Pre-compute weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Initial population
        pops = []
        fitvals = []
        
        for i in range(lambda_):
            z = rng.standard_normal(n)
            y = x + sigma * (B @ (D * z))
            y = np.clip(y, -5, 5)
            fy = func(y)
            pops.append(y)
            fitvals.append(fy)
            
        # Initial best is x0
        # Note: x0 might not be in pops, but we track global best
        # However, CMA-ES usually updates x to the weighted mean.
        # We need to be careful with the first step.
        
        for it in range(budget // lambda_):
            # Sort by fitness
            indices = np.argsort(fitvals)
            pops = [pops[i] for i in indices]
            fitvals = [fitvals[i] for i in indices]
            
            # Update mean
            y_w = np.sum(weights[:, None] * np.array(pops[:mu]), axis=0)
            x_new = y_w
            
            # Update step size
            y_sigma = (B @ (D * np.array(pops[:mu] - x) / sigma))
            norm_y_sigma = np.linalg.norm(y_sigma)
            hsig = (np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1) / mu_eff)) / chi_n)
            ps = (1 - c_sigma) * ps + hsig * np.sqrt(mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)
            
            # Update evolution paths
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (B @ (D * (x_new - x) / sigma))
            
            # Update covariance matrix
            C = (B @ (D ** 2) @ B.T)
            delta_h = (1 - c_c) ** (-0.5) + 2 * (c_c - c_1) / (c_1 + c_mu)
            hpc = np.linalg.norm(pc) < (1.4 + 1 / n) * chi_n * np.sqrt(1 - (1 - c_c) ** (it + 1))
            
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(weights[:, None] * (np.array(pops[:mu] - x) / sigma)[:, None] * (np.array(pops[:mu] - x) / sigma)[None, :], axis=0)
            
            C = (1 + c_1 * (hpc * delta_h * rank_one - rank_one) + c_mu * rank_mu) * C
            
            # Regularize C
            C = 0.5 * (C + C.T)
            
            # Eigen decomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.clip(D, 1e-12, None))
            
            x = x_new
            
            # Generate new population
            new_pops = []
            new_fitvals = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                new_pops.append(y)
                new_fitvals.append(fy)
            
            pops = new_pops
            fitvals = new_fitvals
            
        return x, min(fitvals)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        rng = np.random.default_rng(42)
        
        # Split budget among restarts
        # Lunacek is multimodal. 3-5 restarts is reasonable.
        n_restarts = 5
        budget_per_restart = self.budget // n_restarts
        
        best_x = None
        best_f = np.inf
        
        for r in range(n_restarts):
            if time.time() > deadline - 10: # Check if we have enough time for this restart
                break
            
            # Initialize
            # Start near center or random
            if r == 0:
                x0 = np.zeros(self.dim)
            else:
                x0 = rng.uniform(-4, 4, self.dim)
            
            f0 = func(x0)
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Run CMA-ES
            # Adjust budget for this restart
            # We have used 1 eval for x0.
            remaining_budget = budget_per_restart - 1
            if remaining_budget <= 0:
                continue
                
            # Use a fixed sigma based on domain size
            sigma0 = 2.0
            
            x_best, f_best = self._cmaes(func, x0, f0, sigma0, remaining_budget, rng)
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
                
        # If we still have time and budget, do a final polish or just return
        # The task says return value is ignored, but we return for completeness
        if best_x is None:
             best_x = np.zeros(self.dim)
             
        return best_x, best_f

# EVOLVE-BLOCK-END
