"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for Bent Cigar (f12) with adaptive covariance."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        rng = np.random.default_rng(42)
        dim = self.dim
        n = dim
        budget = self.budget

        # CMA-ES parameters
        # Population size: 4n+2
        lambda_ = 4 * n + 2  # 22 for n=5
        # We have 500 calls total. 1 initial + 500-1 = 499 remaining.
        # 499 // 22 = 22 generations.
        
        # Initial mean
        mean = rng.uniform(-4, 4, n)
        f_mean = func(mean)
        calls = 1

        # Initial standard deviation
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(n)
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Adaptation parameters
        mu = lambda_ // 2  # selection size
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 * mu_eff - 1) / (n + 9)
        c_sigma_inv = 1.0 / c_sigma
        
        # Precompute C^{-1/2}
        # Initially C=I, so C_inv_sqrt = I
        C_inv_sqrt = np.eye(n)
        C_inv_sqrt_eigvals = None
        C_inv_sqrt_eigvecs = None
        eig_update_counter = 0
        eig_update_freq = 10  # update eigendecomp every 10 generations
        
        # Track best
        best_x = mean.copy()
        best_fx = f_mean

        while calls + lambda_ <= budget and time.time() < deadline:
            # Generate population
            offspring = np.zeros((lambda_, n))
            f_offspring = np.zeros(lambda_)
            
            for i in range(lambda_):
                z = rng.standard_normal(n)
                y = C_inv_sqrt @ z
                x_i = mean + sigma * y
                x_i = np.clip(x_i, -5, 5)
                offspring[i] = x_i
                
                if calls + i + 1 > budget:
                    break
                f_offspring[i] = func(x_i)
                calls += 1
                
                if f_offspring[i] < best_fx:
                    best_fx = f_offspring[i]
                    best_x = x_i.copy()
            
            if calls >= budget:
                break

            # Sort offspring by fitness
            order = np.argsort(f_offspring)
            offspring = offspring[order]
            f_offspring = f_offspring[order]
            
            # Select best mu
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += weights[i] * (offspring[i] - mean) / sigma
            
            # Update mean
            mean = mean + sigma * y_w
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_w)
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = (norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**((calls + 1) // lambda_ + 1)) < (1.4 + 2 / (n + 1)) * np.sqrt((mu_eff + 1) / (n + 1)))
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                y_i = (offspring[i] - mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c_c * delta_h - c_c - mu_eff / (n + 1)) * C + c_c * (rank_one + (1 - h_sigma) * delta_h * np.eye(n)) + mu_eff / (n + 1) * rank_mu
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-6, 100.0)
            
            # Update C^{-1/2} periodically
            eig_update_counter += 1
            if eig_update_counter >= eig_update_freq:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip eigenvalues to positive
                eigvals = np.maximum(eigvals, 1e-10)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                eig_update_counter = 0

            # Update mean fitness estimate
            f_mean = np.mean(f_offspring[:mu])

            # If stuck, try a restart or local refinement
            # For Bent Cigar, once close, CMA-ES converges fast.
            # If sigma is very small, we might be at the minimum.
            if sigma < 1e-4:
                break

        # Polish with a few local steps if budget remains
        if calls < budget and time.time() < deadline:
            # Simple local search around best_x
            local_sigma = 0.01
            for _ in range(min(budget - calls, 50)):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + local_sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_fx:
                    best_x = y.copy()
                    best_fx = fy
                    local_sigma *= 1.5
                else:
                    local_sigma *= 0.7

        return best_x, best_fx
# EVOLVE-BLOCK-END
