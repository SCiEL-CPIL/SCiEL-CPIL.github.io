"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Pure CMA-ES optimizer for Sharp Ridge (f13)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        # CMA-ES Parameters
        n = self.dim
        # Population size: (4 + ln(n)) * n
        # For n=5: (4 + 1.6) * 5 = 28
        lambda_pop = int(np.ceil((4 + np.log(n)) * n))
        
        # Initial sigma
        # Box is [-5, 5], center 0. Range 10.
        # Start with sigma covering a good portion of the box
        sigma = 2.0
        
        # Initial mean: center of the box
        x_mean = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        
        # Eigen decomposition cache
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-20))
        
        # Step size control parameters
        mu = lambda_pop // 2  # Weights for top mu individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Evolution path parameters
        chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n ** 2))
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Count function calls
        calls = 0
        best_x = x_mean.copy()
        best_f = np.inf
        
        def time_ok():
            return time.time() < deadline

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return np.inf
            f_val = func(x)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Evaluate initial mean
        best_f = evaluate(x_mean)
        
        # Main CMA-ES Loop
        while calls < self.budget and time_ok():
            # 1. Sample new points
            # Ensure D is not too small/large
            if np.any(D < 1e-15) or np.any(D > 1e15):
                break
                
            # Update D and B if C changed (it does every step, but we reuse B,D from previous eigh)
            # Note: In a full implementation, we update C then eigh. 
            # For efficiency, we sample using current B, D.
            
            x_samples = []
            f_samples = []
            
            for i in range(lambda_pop):
                if calls >= self.budget or not time_ok():
                    break
                z = np.random.normal(0, 1, n)
                x_new = x_mean + sigma * (B @ (D * z))
                x_new = np.clip(x_new, -5, 5)
                
                f_new = evaluate(x_new)
                if np.isinf(f_new):
                    continue
                    
                x_samples.append(x_new)
                f_samples.append(f_new)
                
                # If we ran out of budget in the middle of a generation, break out
                if calls >= self.budget:
                    break
            
            if len(x_samples) < 2:
                break
                
            x_samples = np.array(x_samples)
            f_samples = np.array(f_samples)
            
            # 2. Sort by fitness
            idx = np.argsort(f_samples)
            x_samples = x_samples[idx]
            f_samples = f_samples[idx]
            
            # Take top mu
            x_top = x_samples[:mu]
            f_top = f_samples[:mu]
            
            # 3. Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, np.newaxis] * x_top, axis=0)
            
            # 4. Update covariance
            # Calculate delta mean
            delta_mean = (x_mean - x_mean_old) / sigma
            
            # Update evolution path p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * delta_mean
            
            # Check h_sigma condition
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1 if (norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1) / lambda_pop))) < (1.4 + 2 / (n + 1)) * chi_n else 0
            
            # Update evolution path p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * delta_mean
            
            # Update covariance matrix C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(w_i * (y_i @ y_i))
            
            # y_i = (x_i - x_mean_old) / sigma
            y = (x_top - x_mean_old) / sigma
            
            # Rank-1 update
            rank1_update = np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C
            
            # Rank-mu update
            rank_mu_update = np.sum(weights[:, np.newaxis, np.newaxis] * (y[:, np.newaxis, :] * y[:, np.newaxis, :].transpose(0, 2, 1)), axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank1_update + c_mu * rank_mu_update
            
            # Symmetrize C
            C = (C + C.T) / 2.0
            
            # 5. Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # 6. Update eigen decomposition for next iteration
            # Check for positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.any(eigvals < 0):
                # Fix negative eigenvalues
                eigvals = np.maximum(eigvals, 1e-15)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = (C + C.T) / 2.0
                eigvals, eigvecs = np.linalg.eigh(C)
                
            D = np.sqrt(eigvals)
            B = eigvecs
            
            # Stop if sigma is too small or too large
            if sigma < 1e-15 or sigma > 1e15:
                break
                
            # Optional: Stop if progress is negligible
            if calls > 50 and np.std(f_top) < 1e-10:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
