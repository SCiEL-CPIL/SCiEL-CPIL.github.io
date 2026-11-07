"""CMA-ES optimizer for BBOB f6 (Attractive Sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = None
        best_f = float('inf')

        def check_deadline():
            return time.time() >= deadline

        # Initialize CMA-ES
        # Start at the center of the box
        x_mean = np.zeros(dim)
        sigma = 1.0
        
        # Population size
        # Standard rule: lambda = 4 + floor(3 * log(dim))
        # For dim=5, lambda = 4 + 6 = 10
        lam = 10
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Damping
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        
        try:
            C_chol = np.linalg.cholesky(C)
        except:
            C_chol = np.eye(dim)

        rng = np.random.default_rng(42)

        # Main CMA-ES Loop
        while calls < budget and not check_deadline():
            
            # Generate offspring
            offspring = []
            for i in range(lam):
                if calls >= budget or check_deadline():
                    break
                z = rng.standard_normal(dim)
                try:
                    y = x_mean + sigma * (C_chol @ z)
                except:
                    y = x_mean + sigma * z
                # Clip to box [-5, 5]
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls += 1
                offspring.append((y, fy))
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(offspring) < mu:
                break

            # Sort by fitness
            offspring.sort(key=lambda x: x[1])
            selected = offspring[:mu]

            # Weighted mean of selected
            x_new_mean = sum(weights[i] * selected[i][0] for i in range(mu))
            x_new_mean = np.array(x_new_mean)
            
            # Update paths
            try:
                C_inv_sqrt = np.linalg.cholesky(C_inv)
                C_inv_sqrt_T = C_inv_sqrt.T
            except:
                C_inv_sqrt_T = np.eye(dim)

            diff = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt_T @ diff)
            
            p_sigma_norm = np.linalg.norm(p_sigma)
            # h_sigma check
            # chi_n is the expected norm of a standard normal vector
            # If p_sigma is too large, it indicates a step in a new direction
            h_sigma = 1.0 if p_sigma_norm < (1.4 + 2 / (dim + 1)) * chi_n else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff

            # Update covariance
            delta_h = (1.8 + 2 / (mu_eff + 1)) * (1 - h_sigma)
            rank_one = np.outer(p_c + delta_h * p_c, p_c + delta_h * p_c)
            
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                yi = selected[i][0] - x_mean
                rank_mu += weights[i] * np.outer(yi, yi) / (sigma ** 2)

            C = (1 + c_1 * (rank_one - C) + c_mu * (rank_mu - C)) * C
            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (p_sigma_norm / chi_n - 1))
            
            # Ensure sigma stays reasonable
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update C_inv and C_chol
            x_mean = x_new_mean
            try:
                C_inv = np.linalg.inv(C)
                C_chol = np.linalg.cholesky(C)
            except:
                # Reset if numerical issues
                C = np.eye(dim)
                C_inv = np.eye(dim)
                C_chol = np.eye(dim)
                sigma = 0.5

            # Stagnation check: if sigma is very small, we might be stuck
            # or converged. If we are close to the boundary and sigma is small, 
            # we can stop early to save time, but the loop condition handles budget.
            if sigma < 1e-8:
                break

        # Final check: ensure we return a valid point
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
