"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Single-run CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin (300s limit, 10s buffer)
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Standard CMA-ES settings for dim=5, budget=500
        # lambda = 4 * floor(1 + 3*ln(n)) is standard, but for small n, lambda=4n is often better
        # Let's use lambda = 4 * dim = 20 for dim=5? 
        # Actually, standard is lambda = 4 + floor(3*ln(n)). For n=5, 4+3*1.6=8.8 -> 9? 
        # Let's stick to a robust default: lambda = 4 * dim is common for small dimensions in some implementations, 
        # but the canonical CMA-ES uses lambda = 4 + floor(3*ln(n)). 
        # For n=5, 4 + 4 = 8. Let's use lambda=10 for a bit more exploration per generation.
        # With budget 500 and lambda 10, we get 50 generations. This is plenty for CMA-ES to converge.
        
        lam = 10
        
        # Initial parameters
        # Start near the center, but slightly off to avoid symmetry issues if any
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, roughly 1/5 of the box width
        
        # CMA-ES internal variables
        C = np.eye(dim)  # Covariance matrix
        C_inv = np.eye(dim)  # Inverse covariance
        C_sqrt = np.eye(dim)  # Square root of C
        C_inv_sqrt = np.eye(dim)  # Inverse square root of C
        
        # Evolution path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Adaptation rates
        mu = lam // 2  # Weights for the best mu individuals
        # Standard weights: w_i = ln(mu + 0.5) - ln(i)
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= weights.sum()
        
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        best_x = x_mean.copy()
        best_f = float('inf')
        used_calls = 0
        
        # Initial evaluation
        f_mean = func(x_mean)
        used_calls += 1
        if f_mean < best_f:
            best_f = f_mean
            best_x = x_mean.copy()
            
        # Main loop
        while used_calls < budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            # We need to generate lam points
            # Check if we have enough budget for a full generation
            if budget - used_calls < lam:
                # If not enough for full generation, do a small batch or break
                # To maximize score, we should use remaining calls.
                # However, CMA-ES relies on full generations for matrix updates.
                # If we don't have enough, we can just do random local search or break.
                # Given 500 calls and lam=10, we have exactly 50 generations. 
                # So this branch is unlikely unless we start late.
                # If we are here, just do a simple (1+1) step to use the call.
                y = np.clip(x_mean + sigma * np.random.default_rng(0).standard_normal(dim), -5, 5)
                fy = func(y)
                used_calls += 1
                if fy < f_mean:
                    x_mean = y.copy()
                    f_mean = fy
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                continue
            
            # Generate lam samples from N(x_mean, sigma^2 * C)
            # We use a local RNG for reproducibility within the run, but seeded by global state?
            # The task says "hidden seeded shift", but our optimizer can use its own RNG.
            # Using a fixed seed for the optimizer is fine as the problem instances are fixed.
            rng = np.random.default_rng(0) # Resetting seed each call ensures determinism across instances
            
            z = rng.standard_normal((lam, dim))
            # Transform to x space: x = x_mean + sigma * C_sqrt @ z
            # Note: C_sqrt is the matrix such that C = C_sqrt @ C_sqrt.T
            # We maintain C_sqrt via Cholesky or similar. 
            # Standard CMA-ES uses the update: C = C + c_1*(p_c p_c^T - C) + c_mu * sum w_i (y_i y_i^T - C)
            # Then compute new C_sqrt.
            
            # For simplicity and robustness in a short script, we can use the "Rank-One" and "Rank-Mu" updates
            # and recompute the Cholesky factorization every generation.
            
            offspring = x_mean + sigma * (z @ C_sqrt.T)
            
            # Evaluate offspring
            f_offspring = np.zeros(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                x_i = np.clip(offspring[i], -5, 5)
                f_i = func(x_i)
                f_offspring[i] = f_i
                used_calls += 1
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
            
            if used_calls >= budget:
                break
                
            # Sort offspring by fitness
            order = np.argsort(f_offspring)
            y = offspring[order]  # Sorted offspring
            f_sorted = f_offspring[order]
            
            # Update mean
            x_mean_new = x_mean + sigma * np.sum(weights[:, None] * y[:mu], axis=0)
            
            # Update evolution path p_c
            # p_c = (1 - c_c) * p_c + c_c * sqrt(mu_eff) * C_inv_sqrt @ (x_mean_new - x_mean) / sigma
            # Note: (x_mean_new - x_mean) / sigma is the weighted sum of z transformed by C_sqrt
            # Actually, the standard update is:
            # p_c = (1 - c_c) * p_c + c_c * sqrt(mu_eff) * C_inv_sqrt @ (x_mean_new - x_mean)
            
            # Calculate C_inv_sqrt @ (x_mean_new - x_mean)
            # We have C_inv_sqrt.
            # delta_mean = (x_mean_new - x_mean)
            # p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (C_inv_sqrt @ delta_mean)
            
            # Wait, the standard CMA-ES update for p_c uses the "untransformed" step?
            # No, it uses the step in the transformed space.
            # Let's stick to the standard formulas from Hansen's paper.
            
            # 1. Update mean
            # x_mean = x_mean + sigma * sum(w_i * y_i)
            
            # 2. Update p_c
            # p_c = (1 - c_c) p_c + c_c * sqrt(mu_eff) * C^{-1/2} (x_{new} - x_{old})
            # Here (x_{new} - x_{old}) = sigma * sum(w_i * y_i)
            # So C^{-1/2} (x_{new} - x_{old}) = sigma * C^{-1/2} C^{1/2} sum(w_i * z_i) = sigma * sum(w_i * z_i)
            # This simplifies things! We don't need to multiply by C_inv_sqrt if we use the z-vectors directly?
            # Let's verify.
            # y_i = C^{1/2} z_i.
            # x_new - x_old = sigma * sum w_i C^{1/2} z_i.
            # C^{-1/2} (x_new - x_old) = sigma * sum w_i z_i.
            # So p_c update is:
            # p_c = (1 - c_c) p_c + c_c * sqrt(mu_eff) * sigma * sum(w_i * z_sorted_i)
            
            # Let's use this simplified update for p_c.
            z_sorted = z[order]
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * sigma * np.sum(weights[:, None] * z_sorted[:mu], axis=0)
            
            # 3. Update p_sigma
            # p_sigma = (1 - c_sigma) p_sigma + c_sigma * sqrt(mu_eff) * C^{-1/2} (x_{new} - x_{old}) / sigma
            # Again, C^{-1/2} (x_{new} - x_{old}) / sigma = sum(w_i * z_sorted_i)
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * np.sum(weights[:, None] * z_sorted[:mu], axis=0)
            
            # 4. Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / chi_n - 1))
            # d_sigma = 1 + 2 * max(0, sqrt((mu_eff - 1) / (dim + 1)) - 1)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1)
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # 5. Update C
            # h_sig = (||p_sigma|| / sqrt(1 - (1 - c_sigma)^{2*(gen+1)})) / chi_n < 1.4 + 2/(dim+1)
            # We approximate the denominator as 1 for simplicity in short runs, or just check norm.
            # For robustness, let's just update C directly.
            
            # Rank-one update
            # C = C + c_1 * ( (1 + (1 - c_c) h_sig) p_c p_c^T - C )
            # Let's assume h_sig = 1 for now (conservative)
            h_sig = 1.0
            C = C + c_1 * ((1 + (1 - c_c) * h_sig) * np.outer(p_c, p_c) - C)
            
            # Rank-mu update
            # C = C + c_mu * sum w_i (y_i y_i^T - C)
            # y_i are the sorted steps in x-space? No, y_i are the steps in the transformed space?
            # In CMA-ES, y_i = x_i - x_mean_old.
            # The update is C = C + c_mu * sum w_i ( (y_i/sigma) (y_i/sigma)^T - C )
            # (y_i/sigma) = C^{1/2} z_i
            # So (y_i/sigma) (y_i/sigma)^T = C^{1/2} z_i z_i^T C^{1/2}
            
            # This requires C^{1/2}. We have C_sqrt.
            # Let's compute the weighted sum of outer products of the normalized steps.
            # normalized_steps = y[:mu] / sigma  # This is C^{1/2} z_sorted[:mu]
            # We can compute this as (z_sorted[:mu] @ C_sqrt.T)
            
            # Actually, it's easier to just use the x-space steps.
            # y_x = offspring[order][:mu] - x_mean
            # normalized_y = y_x / sigma
            # C = C + c_mu * sum w_i ( outer(normalized_y_i, normalized_y_i) - C )
            
            y_x = y[:mu]
            normalized_y = y_x / sigma
            for i in range(mu):
                C += c_mu * weights[i] * (np.outer(normalized_y[i], normalized_y[i]) - C)
            
            # Ensure C is symmetric
            C = (C + C.T) / 2.0
            
            # Compute new C_sqrt and C_inv_sqrt
            # Use Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
                C_sqrt = L
                C_inv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # Fallback if C becomes non-positive-definite
                # Add small regularization
                C += 1e-6 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                    C_sqrt = L
                    C_inv_sqrt = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    # Reset to identity if all else fails
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)
                    sigma = 2.0
                    p_c = np.zeros(dim)
                    p_sigma = np.zeros(dim)
                    
            x_mean = x_mean_new
            f_mean = f_sorted[0]  # Best offspring is the new mean's fitness? No, mean is a new point.
            # Actually, x_mean is the new center. We don't evaluate it immediately in standard CMA-ES,
            # but we can keep track of the best found so far.
            
            # Check for convergence
            if sigma < 1e-12:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
