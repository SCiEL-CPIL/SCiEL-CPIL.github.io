import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for continuous black-box functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        
        # Learning rates
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        
        # Initial state
        rng = np.random.default_rng(42)
        
        # Start near center to avoid boundary issues initially, but random is also fine
        # Bent cigar is usually centered at 0 in standard BBOB, but here it's shifted.
        # A random start is safer for unknown shifts.
        x_mean = rng.uniform(-4, 4, dim)
        sigma = 2.0
        
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_fx = func(x_mean)
        budget_used = 1
        
        def update_cholesky(C):
            """Compute square root and inverse square root of C."""
            try:
                L = np.linalg.cholesky(C)
                C_sqrt = L
                C_inv = np.linalg.inv(C)
                # Inverse square root via Cholesky of inverse
                L_inv = np.linalg.cholesky(C_inv)
                C_inv_sqrt = np.linalg.inv(L_inv).T
            except np.linalg.LinAlgError:
                # Fallback if matrix becomes singular
                C = np.eye(dim)
                L = np.eye(dim)
                C_sqrt = L
                C_inv = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
            return C_sqrt, C_inv, C_inv_sqrt

        C_sqrt, C_inv, C_inv_sqrt = update_cholesky(C)

        while budget_used < budget:
            if time.time() >= deadline:
                break

            # Generate population
            offspring = []
            offspring_f = []
            
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C_sqrt @ z)
                
                # Clip to bounds
                # Reflection strategy is better for CMA-ES to maintain distribution shape
                # but simple clipping is safer and often sufficient for box constraints
                # Let's use clipping for simplicity and robustness
                y_clipped = np.clip(y, -5, 5)
                
                # Check if it's a duplicate of a previously evaluated point in this generation
                # to avoid wasting budget (though func is deterministic, duplicates don't add info)
                # For simplicity, we just evaluate.
                
                fy = func(y_clipped)
                offspring.append(y_clipped)
                offspring_f.append(fy)
                budget_used += 1
                
                if budget_used >= budget:
                    break

            if not offspring:
                break

            offspring = np.array(offspring)
            offspring_f = np.array(offspring_f)

            # Track best
            idx_min = np.argmin(offspring_f)
            if offspring_f[idx_min] < best_fx:
                best_fx = offspring_f[idx_min]
                best_x = offspring[idx_min].copy()

            # Sort offspring by fitness
            order = np.argsort(offspring_f)
            offspring_sorted = offspring[order]
            offspring_f_sorted = offspring_f[order]

            # Select top mu
            y = offspring_sorted[:mu]
            
            # Update mean
            x_mean_new = np.sum(weights[:, None] * y, axis=0)
            
            # Update evolution paths
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C_inv_sqrt @ (x_mean_new - x_mean)
            # pc = (1 - c_c) * pc + sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x_mean_new - x_mean)
            
            d = (x_mean_new - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ d)
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * d
            
            # Update covariance matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (||ps||^2 / chi_n^2 - 1) * C) + c_mu * sum(weights * (y_i - x_mean) @ (y_i - x_mean).T)
            
            diff = y - x_mean
            outer_sum = np.sum(weights[:, None, None] * (diff[:, :, None] * diff[:, None, :]), axis=0)
            
            ps_norm_sq = np.dot(ps, ps)
            c_update = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (ps_norm_sq / chi_n**2 - 1) * C) + c_mu * outer_sum
            
            # Ensure symmetry
            c_update = (c_update + c_update.T) / 2
            
            # Regularize if necessary
            # If eigenvalues are too small, add identity
            eigvals = np.linalg.eigvalsh(c_update)
            if np.min(eigvals) < 1e-12:
                c_update += 1e-12 * np.eye(dim)
                
            C = c_update
            x_mean = x_mean_new
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update matrix factorizations
            C_sqrt, C_inv, C_inv_sqrt = update_cholesky(C)

            # Check for convergence
            if sigma < 1e-10:
                break

        return best_x, best_fx
