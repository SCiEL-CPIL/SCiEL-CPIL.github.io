"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with time management."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Initial point
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        
        # CMA-ES parameters
        sigma = 2.0
        pop_size = 2 * (4 + int(np.log(dim)))
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        step_size = sigma
        
        # Strategy parameters
        alpha_mu = 0.54 + 0.38 * (dim / (dim + 2))
        c_sigma = (dim + 1.7) / (dim + 9.5 + alpha_mu)
        c_c = (4 + dim / dim) / (dim + 4.1 + alpha_mu)
        c_1 = 2 / ((dim + 1.3)**2 + alpha_mu)
        c_mu = min(1 - c_1, 2 * (alpha_mu - 0.9) / ((dim + 2)**2 + alpha_mu))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Weights
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        
        calls_made = 1
        
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            offspring = []
            offspring_f = []
            
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                # Sample from multivariate normal using C, B, D
                z = rng.standard_normal(dim)
                y = step_size * (B @ (D * z))
                y = np.clip(x + y, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append(y)
                offspring_f.append(fy)
            
            if len(offspring) == 0:
                break
            
            # Sort offspring
            idx = np.argsort(offspring_f)
            offspring = [offspring[i] for i in idx]
            offspring_f = [offspring_f[i] for i in idx]
            
            # Update best
            if offspring_f[0] < best_f:
                best_f = offspring_f[0]
                best_x = offspring[0].copy()
            
            # Update mean
            new_x = sum(w * offspring[i] for i, w in enumerate(weights))
            
            # Update sigma
            y_mean = (new_x - x) / step_size
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            sigma_factor = np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - chi_n))
            step_size = np.clip(step_size * sigma_factor, 1e-10, 10)
            
            # Update covariance
            y_w = (new_x - x)
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(y_w, y_w) + (1 - (1 - c_1 - c_mu) * (1 - (1 - c_1) / (c_1 * dim))) * np.eye(dim))
            
            # Update B and D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            B = eigvecs
            D = np.sqrt(eigvals)
            
            x = new_x
            fx = best_f
            
            # Check convergence
            if step_size < 1e-15:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
