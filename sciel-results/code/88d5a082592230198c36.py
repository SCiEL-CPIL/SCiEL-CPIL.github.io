"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for Sphere function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Standard CMA-ES initialization
        # x0: center of the box seems reasonable given the hidden shift is in [-4, 4]
        x_mean = np.zeros(d)
        
        # Initial sigma: box is [-5, 5], width 10.
        # A common heuristic is sigma = 0.25 * (max - min) or similar.
        # 2.0 is a good start to explore the 10-width box.
        sigma = 2.0
        
        # Population size
        # Standard lambda = 4 + floor(3 ln d)
        lam = 4 + int(3 * np.log(d))
        
        # Step size control parameters
        mu = lam // 2  # Number of best points for update
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Memory
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        C = np.eye(d)
        
        # Initial evaluation
        x0 = x_mean
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls = 1
        
        # Precompute matrix square root of C? Not needed for sampling, just C itself.
        # We need to sample from N(0, C).
        # Use Cholesky or SVD. SVD is more stable for near-singular C.
        
        def sample_normal(sigma_val, C_mat):
            # Sample from N(0, C_mat)
            # Use SVD for stability
            U, S, Vt = np.linalg.svd(C_mat)
            # C = U S Vt
            # We want z such that z ~ N(0, C)
            # z = U * sqrt(S) * r, where r ~ N(0, I)
            r = np.random.randn(d)
            z = U @ (np.sqrt(S) * r)
            return sigma_val * z

        # Main loop
        while calls < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break
                
            # Generate population
            x_pop = np.zeros((lam, d))
            f_pop = np.zeros(lam)
            
            for i in range(lam):
                # Sample direction
                # Note: In standard CMA-ES, we sample z ~ N(0, C), then x = x_mean + sigma * z
                # But we must clip to box [-5, 5]. Clipping distorts the distribution.
                # For Sphere, this is usually acceptable if sigma is small enough or if we handle boundary carefully.
                # Given the box is tight, we might need to reduce sigma if we hit boundaries often.
                
                z = sample_normal(1.0, C)
                x_i = x_mean + sigma * z
                # Clip to box
                x_i = np.clip(x_i, -5, 5)
                
                x_pop[i] = x_i
                f_i = func(x_i)
                f_pop[i] = f_i
                calls += 1
                
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
                
                if time.time() > deadline - 5.0:
                    break
            
            if calls >= budget:
                break
                
            # Sort by fitness
            idx = np.argsort(f_pop)
            x_sorted = x_pop[idx]
            f_sorted = f_pop[idx]
            
            # Update mean
            x_old_mean = x_mean.copy()
            x_mean = np.sum(weights[:, np.newaxis] * x_sorted[:mu], axis=0)
            
            # Update sigma
            # p_sigma update
            # h_sigma check is omitted for simplicity/robustness, standard CMA-ES often works without it for small d
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C @ (x_mean - x_old_mean) / sigma)
            
            # Chi-square distribution constant
            chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
            
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1))) < (1.4 + 2 / (d + 1)) * chi_d:
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / chi_d - 1))
            
            # Update C
            # p_c update
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_old_mean) / sigma
            
            # Rank-one update
            y_sigma = (x_mean - x_old_mean) / sigma if sigma > 0 else np.zeros(d)
            C = (1 + c_1 * (c_mu * (np.outer(y_sigma, y_sigma) - C) - c_mu) + c_mu) * C \
                + c_1 * (np.outer(p_c, p_c) - C)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            
            # Regularize if eigenvalues become too small or negative
            # Simple check: trace or min eigenvalue
            try:
                eigvals = np.linalg.eigvalsh(C)
                if np.any(eigvals < 1e-12):
                    # Add small regularization
                    C += 1e-10 * np.eye(d)
            except:
                C = np.eye(d)
                sigma = 1.0 # Reset if something goes wrong? Maybe too drastic.
                
            # Optional: Reset if stuck?
            # If sigma becomes very small, we might be stuck in local minimum or converged.
            # For Sphere, convergence is fast.
            if sigma < 1e-10:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
