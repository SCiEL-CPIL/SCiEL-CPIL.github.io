"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES optimizer for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Global best tracking
        best_x = None
        best_f = np.inf
        
        # CMA-ES Parameters
        # For dim=5, lambda=10 is a good balance
        lam = 10
        mu = lam // 2
        
        # Weights for the mu best points
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # State variables for CMA-ES
        # We will implement a restart mechanism
        
        evals_used = 0
        
        while evals_used < budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Initialize or re-initialize CMA-ES state
            if best_x is None:
                x_center = np.random.uniform(-4, 4, dim)
                best_x = x_center.copy()
                best_f = func(best_x)
                evals_used += 1
            else:
                # Restart logic: If we are restarting, we use the current best as center
                # This ensures we don't lose the global best found so far
                x_center = best_x.copy()
                
            sigma = 2.0 # Initial step size
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 + 1/(4*dim) - 1/(20*dim**2))
            
            stagnation_count = 0
            last_best_f = best_f
            
            # Inner loop for one CMA-ES run
            while evals_used < budget:
                if time.time() > deadline:
                    break
                
                # Generate population
                # Sample from N(x_center, sigma^2 * C)
                # Efficient sampling: z ~ N(0, I), y = C^0.5 z
                # For small dim, we can just use np.random.multivariate_normal or Cholesky
                # Since dim is small (5), Cholesky of C is cheap
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim) # Fallback if C becomes singular
                    
                z = np.random.randn(lam, dim)
                # y = z @ L.T
                y = z @ L.T
                population = x_center + sigma * y
                
                # Clip to box [-5, 5]
                population = np.clip(population, -5, 5)
                
                # Evaluate
                fitnesses = np.array([func(x) for x in population])
                evals_used += lam
                
                # Check for improvements
                idx_min = np.argmin(fitnesses)
                if fitnesses[idx_min] < best_f:
                    best_f = fitnesses[idx_min]
                    best_x = population[idx_min].copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                
                # Sort population by fitness
                order = np.argsort(fitnesses)
                y_sorted = population[order]
                f_sorted = fitnesses[order]
                
                # Update center
                x_new = np.sum(weights[:, np.newaxis] * y_sorted[:mu], axis=0)
                
                # Update evolution paths
                pc = (1 - c_c) * pc + c_c * (x_new - x_center) / sigma
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c)**(2 * (evals_used % 100 + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
                ps = (1 - c_sigma) * ps + hsig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x_new - x_center) / sigma
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update covariance
                # Rank one update
                v = np.outer(pc, pc)
                # Rank mu update
                y_diff = (y_sorted[:mu] - x_center) / sigma
                y_w = weights[:, np.newaxis] * y_diff
                rank_mu = np.sum(y_w * y_w.T, axis=0)
                
                C = (1 - c_1 - c_mu) * C + c_1 * (v + (1 - hsig) * c_c * (2 - c_c) * np.outer(pc, pc)) + c_mu * rank_mu
                
                # Symmetrize to prevent numerical drift
                C = (C + C.T) / 2
                
                x_center = x_new
                
                # Restart conditions
                # 1. Stagnation: No improvement for 30 generations (300 evals)
                # 2. Convergence: Sigma becomes very small
                # 3. Max generations for a single run (to ensure diversity)
                if stagnation_count > 30 or sigma < 1e-8 or (evals_used % lam == 0 and (evals_used // lam) > 50):
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
