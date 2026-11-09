"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Standard CMA-ES parameters
        lam = int(4 + 3 * np.log(dim))
        
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights ** 2)
        
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Initial parameters
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        
        # Precompute C_inv_sqrt and C_sqrt
        # C = I initially
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # RNG
        rng = np.random.default_rng(42)
        
        # Initial evaluation
        f_best = func(x_mean)
        x_best = x_mean.copy()
        n_evals = 1
        
        # Main loop
        while n_evals < budget:
            # Determine population size
            current_lam = lam
            if n_evals + current_lam > budget:
                current_lam = budget - n_evals
                if current_lam <= 0:
                    break
            
            # Adjust mu and weights for current population size
            mu_curr = min(mu, current_lam)
            if mu_curr == 0:
                break
            w_curr = np.log(mu_curr + 1) - np.log(np.arange(1, mu_curr + 1))
            w_curr /= np.sum(w_curr)
            mueff_curr = 1 / np.sum(w_curr ** 2)
            
            # Sample population
            z = rng.standard_normal((current_lam, dim))
            pop = x_mean + sigma * (z @ C_sqrt.T)
            pop = np.clip(pop, -5, 5)
            
            # Evaluate population
            f_pop = np.array([func(x) for x in pop])
            n_evals += current_lam
            
            # Sort by fitness
            indices = np.argsort(f_pop)
            pop_sorted = pop[indices]
            f_sorted = f_pop[indices]
            
            # Update best
            if f_sorted[0] < f_best:
                f_best = f_sorted[0]
                x_best = pop_sorted[0].copy()
            
            # Save old x_mean
            x_mean_old = x_mean.copy()
            
            # Update x_mean
            x_mean = np.sum(w_curr[:, np.newaxis] * pop_sorted[:mu_curr], axis=0)
            
            # Update evolution paths
            # p_sigma = (1 - cs) * p_sigma + sqrt(cs * (2 - cs) * mueff_curr) * C_inv_sqrt @ (x_mean - x_mean_old) / sigma
            if sigma > 1e-16:
                diff = (x_mean - x_mean_old) / sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff_curr) * (C_inv_sqrt @ diff)
                
                # Update sigma
                # sigma = sigma * exp(cs / (dim * (1 + cs)) * (||p_sigma||^2 / dim - 1))
                # Note: The standard formula uses (||p_sigma||^2 / dim - 1)
                # But there is a correction term for the expected norm of p_sigma
                # Expected norm squared of p_sigma is dim * (1 - (1 - cs)^2)
                # Let's use the standard formula:
                # sigma *= exp( (cs / (dim * (1 + cs))) * ( (p_sigma @ p_sigma) / dim - 1 ) )
                # Actually, the correct formula is:
                # sigma *= exp( (cs / (dim * (1 + cs))) * ( (p_sigma @ p_sigma) / dim - (1 - (1 - cs)**2) ) )
                # Let's use the simpler one which is often sufficient:
                # sigma *= exp( (cs / (dim * (1 + cs))) * ( (p_sigma @ p_sigma) / dim - 1 ) )
                
                norm_p_sigma_sq = p_sigma @ p_sigma
                sigma *= np.exp((cs / (dim * (1 + cs))) * (norm_p_sigma_sq / dim - 1))
                
                # Update p_c
                # p_c = (1 - cc) * p_c + h_sigma * sqrt(cc * (2 - cc) * mueff_curr) * C_inv_sqrt @ (x_mean - x_mean_old) / sigma
                # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1 - cs)^t) < (1.4 + 2/(dim+1)) * sqrt((dim+1)(1 - (1 - cs)^t))
                # For simplicity, we can approximate h_sigma = 1 or use a fixed threshold
                # Let's use a simple heuristic: h_sigma = 1 if norm_p_sigma_sq < 2 * dim else 0
                # Or better, use the standard test:
                # h_sigma = 1 if norm_p_sigma_sq / dim < (1.4 + 2/(dim+1))**2 * (1 - (1 - cs)**2) else 0
                # Let's approximate (1 - cs)**t as 0 for large t, so threshold is (1.4 + 2/(dim+1))**2
                threshold = (1.4 + 2 / (dim + 1)) ** 2
                h_sigma = 1.0 if norm_p_sigma_sq / dim < threshold else 0.0
                
                p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mueff_curr) * (C_inv_sqrt @ diff)
            
            # Update covariance matrix
            # C = (1 - c1 - cmu) * C + c1 * (p_c @ p_c + (1 - h_sigma) * cc * (2 - cc) * C) + cmu * sum(w_i * (y_i - x_mean) @ (y_i - x_mean))
            # where y_i = (pop_sorted[i] - x_mean_old) / sigma
            
            # Calculate the weighted sum of outer products
            # y_i = (pop_sorted[i] - x_mean_old) / sigma
            if sigma > 1e-16:
                y = (pop_sorted[:mu_curr] - x_mean_old) / sigma
                # Outer products
                # y is (mu_curr, dim)
                # We need sum(w_i * y_i @ y_i)
                # This is (w_curr[:, np.newaxis] * y) @ y.T
                weighted_outer = (w_curr[:, np.newaxis] * y) @ y.T
            else:
                weighted_outer = np.zeros((dim, dim))
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * cc * (2 - cc) * C) + cmu * weighted_outer
            
            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2
            
            # Update C_sqrt and C_inv_sqrt
            # Use eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to be positive
            eigvals = np.maximum(eigvals, 1e-16)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
            
            # Optional: reset sigma if it becomes too large or small
            if sigma > 100:
                sigma = 100
            if sigma < 1e-6:
                sigma = 1e-6
        
        return x_best, f_best
# EVOLVE-BLOCK-END
