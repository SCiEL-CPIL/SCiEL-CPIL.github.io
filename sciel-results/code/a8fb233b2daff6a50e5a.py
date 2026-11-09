import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        n = self.dim
        # CMA-ES parameters
        # lambda = 4 + floor(3 * ln(n))
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2  # number of parents
        
        # Initial state
        x_mean = np.zeros(n)
        sigma = 2.0
        
        # Covariance matrix C (identity)
        C = np.eye(n)
        # Inverse of C
        C_inv = np.eye(n)
        # Cholesky factor of C
        C_chol = np.eye(n)
        # Inverse of Cholesky
        C_chol_inv = np.eye(n)
        
        # Step size control parameters
        cs = (mu + 2) / (n + mu + 5)
        ds = 1 + 2 * max(0, sqrt(cs - 1 / (n + 1))) + cs
        c1 = 2 / ((n + 1.3) ** 2 + mu)
        cmu = min(1 - c1, 2 * ((mu + 0.5) / (n + 2) ** 2 - 1))
        
        # State variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = x_mean
        best_f = func(x_mean)
        calls = 1
        
        # Pre-generate random numbers? No, use rng.
        rng = np.random.default_rng(42)
        
        while calls < self.budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            # We need to generate lam samples
            # To save time, we can generate them in a loop or vectorized
            # Vectorized is faster
            
            # Sample z from standard normal
            # z shape: (lam, n)
            z = rng.standard_normal((lam, n))
            
            # Transform to y = sigma * C_chol @ z
            # y = z @ C_chol.T * sigma
            y = (z @ C_chol.T) * sigma
            # Offspring x_i = x_mean + y_i
            x_pop = x_mean[None, :] + y
            
            # Evaluate
            f_pop = np.array([func(xi) for xi in x_pop])
            calls += lam
            
            # Update best
            min_idx = np.argmin(f_pop)
            if f_pop[min_idx] < best_f:
                best_f = f_pop[min_idx]
                best_x = x_pop[min_idx].copy()
                
            # Sort by fitness
            sorted_idx = np.argsort(f_pop)
            x_sorted = x_pop[sorted_idx]
            f_sorted = f_pop[sorted_idx]
            
            # Weighted mean of the best mu
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            w_sum = np.sum(weights)
            w_max = weights[0]
            
            x_new_mean = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update sigma (step size)
            z_mean = np.sum(weights[:, None] * (x_sorted[:mu] - x_mean) / sigma, axis=0)
            # Actually, z_mean should be the mean of the standardized steps
            # y_sorted = x_sorted[:mu] - x_mean
            # z_sorted = y_sorted / sigma
            # z_mean = sum(w * z_sorted)
            
            # Correction for the norm
            # mu_eff = 1 / sum(w^2)
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            # Update p_sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (
                np.linalg.norm(z_mean) / np.sqrt(n) - 1.0
            )
            # Actually, the formula is:
            # p_sigma = (1-cs)*p_sigma + sqrt(cs*(2-cs)*mu_eff) * ( ||z_mean|| / chi_n - 1 )
            # chi_n is the expected norm of a standard normal vector
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (
                np.linalg.norm(z_mean) / chi_n - 1.0
            )
            
            # Update sigma
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update p_c
            # p_c = (1-c1)*p_c + sqrt(c1*(2-c1)*mu_eff) * (x_new_mean - x_mean) / sigma
            # Wait, the term is (x_new_mean - x_mean) / sigma
            # But x_new_mean - x_mean is the step in x-space.
            # We need to transform it back to z-space?
            # The standard CMA-ES update for p_c is:
            # p_c = (1-c1)*p_c + sqrt(c1*(2-c1)*mu_eff) * (x_new_mean - x_mean) / sigma
            # This is correct.
            
            p_c = (1 - c1) * p_c + np.sqrt(c1 * (2 - c1) * mu_eff) * (
                (x_new_mean - x_mean) / sigma
            )
            
            # Update C
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-cs)^{2(gen+1)}) < (1.4 + 2/(n+1)) * chi_n
            # We don't track generations exactly, but we can approximate or just use a simple check.
            # For simplicity and speed, we often skip the h_sigma check or use a fixed threshold.
            # Let's implement the full check to be robust.
            # We need to track the number of generations or use a running average.
            # Since we have a fixed budget, we can just use a simple heuristic:
            # h_sigma = 1.0 if np.linalg.norm(p_sigma) < 1.5 * chi_n else 0.0
            
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < 1.5 * chi_n else 0.0
            
            # Delta term
            # delta = h_sigma * c1 * (1 - c1)
            # Actually, the update is:
            # C = (1 + c1 * h_sigma * (2 - c1) - c1 - cmu) * C
            #    + c1 * p_c * p_c.T
            #    + cmu * sum(w_i * (y_i - y_mean) * (y_i - y_mean).T)
            # where y_i are the steps from the old mean.
            
            # Steps from old mean
            y_steps = x_sorted[:mu] - x_mean[None, :]
            
            # Rank-mu update
            # y_mean is the weighted mean of the steps
            y_mean = np.sum(weights[:, None] * y_steps, axis=0)
            
            # C update
            # Factor 1
            factor_c = 1 + c1 * h_sigma * (2 - c1) - c1 - cmu
            C = factor_c * C
            C += c1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            # sum(w_i * (y_i - y_mean) * (y_i - y_mean).T)
            # This is the covariance of the selected steps, weighted.
            # We can compute this as:
            # diff = y_steps - y_mean[None, :]
            # rank_mu = sum(w_i * diff_i * diff_i.T)
            # To be efficient, we can do:
            # rank_mu = (diff.T * weights) @ diff
            # diff shape: (mu, n), weights shape: (mu,)
            # (diff.T * weights) shape: (n, mu)
            # (n, mu) @ (mu, n) -> (n, n)
            
            diff = y_steps - y_mean[None, :]
            rank_mu = (diff.T * weights) @ diff
            C += cmu * rank_mu
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update Cholesky and inverse
            # This is the expensive part.
            # We need to update C_chol and C_chol_inv.
            # We can use scipy.linalg.cholesky and scipy.linalg.inv
            # Or use the Sherman-Morrison-Woodbury formula for rank updates?
            # Given the small dimension (5), full re-computation is fast enough.
            
            try:
                C_chol = np.linalg.cholesky(C)
                C_chol_inv = np.linalg.inv(C_chol)
            except np.linalg.LinAlgError:
                # If C is not positive definite, add a small identity
                C += 1e-6 * np.eye(n)
                C = (C + C.T) / 2
                C_chol = np.linalg.cholesky(C)
                C_chol_inv = np.linalg.inv(C_chol)
                
            # Update mean
            x_mean = x_new_mean
            
        return best_x, best_f
