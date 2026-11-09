"""CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
import numpy as np


class Optimizer:
    """CMA-ES based continuous optimizer for BBOB f18."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(n))
        # Selection size (top mu/2 for weights)
        mu = lam // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        # Variance reduction factor
        mu_eff = 1.0 / (weights @ weights)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Damping for step size
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Initial step size
        sigma = 2.0
        # Initial mean (center of the box)
        mean = np.zeros(n)
        # Initial covariance matrix (identity)
        C = np.eye(n)
        
        # Pre-compute Cholesky factor for sampling
        # C = B B^T
        B = np.linalg.cholesky(C)
        
        best_x = mean.copy()
        best_f = np.inf
        calls_made = 0
        
        # Evolution path
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Normalization constant for expected norm of N(0, I)
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (256 * n**2))
        
        def sample():
            z = rng.standard_normal(n)
            return mean + sigma * (B @ z)

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        while calls_made < budget:
            # 1. Sample population
            pop = np.empty((lam, n))
            fvals = np.empty(lam)
            
            for i in range(lam):
                if calls_made >= budget:
                    break
                x = sample()
                x = clip(x)
                pop[i] = x
                fvals[i] = func(x)
                calls_made += 1
                
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = x.copy()
            
            if calls_made >= budget:
                break
            
            # 2. Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]
            
            # 3. Update mean
            old_mean = mean.copy()
            mean = weights @ pop_sorted[:mu]
            
            # 4. Update evolution paths and step size
            # Standardize difference
            diff = (mean - old_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
            p_sigma_norm_sq = np.dot(p_sigma, p_sigma)
            
            # Update step size (CSA)
            if p_sigma_norm_sq > 1e-32:
                norm_p_sigma = np.sqrt(p_sigma_norm_sq)
                h_sigma = (norm_p_sigma / chi_n) < (1.4 + 2 / (n + 1))
                if h_sigma:
                    sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                    p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff
                else:
                    p_c = (1 - c_c) * p_c
            
            # 5. Update covariance matrix
            # Rank-1 update
            if p_sigma_norm_sq > 1e-32:
                C += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * (2 - c_c) * C)
            else:
                C += c_1 * (2 - c_c) * C
            
            # Rank-mu update
            y = (pop_sorted[:mu] - old_mean) / sigma
            C += c_mu * (weights @ (y.T * y))
            
            # Ensure positive definiteness and update Cholesky
            # Regularize if needed
            C = 0.5 * (C + C.T)
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not positive definite, add small regularization
                C += 1e-6 * np.eye(n)
                B = np.linalg.cholesky(C)

        return best_x, best_f
