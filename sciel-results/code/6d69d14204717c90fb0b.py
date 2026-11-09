"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for global optimization on BBOB f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        dim = self.dim
        lam = 4 + int(3 * np.log(dim))  # Population size: 12 for dim=5
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Initial point and sigma
        x = rng.uniform(-2, 2, dim)  # Start near center as Rastrigin min is often shifted but within [-4,4]
        sigma = 2.0  # Initial step size to cover a good portion of [-5, 5]
        
        # State variables
        ps = np.zeros(dim)
        pc = np.zeros((dim, dim))
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        best_x = x.copy()
        best_f = func(x)
        calls_made = 1
        
        # Precompute for efficiency
        # We will use a simple loop for budget
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
                
            # Generate population
            # Update C square root approximation (simplified for speed, using Cholesky or SVD)
            # For dim=5, Cholesky is fast
            try:
                L = np.linalg.cholesky(C)
                C_sqrt = L
                C_sqrt_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # Fallback if C is not positive definite (shouldn't happen often)
                C += 1e-10 * np.eye(dim)
                L = np.linalg.cholesky(C)
                C_sqrt = L
                C_sqrt_inv = np.linalg.inv(L)
            
            # Generate samples
            samples = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                samples[i] = x + sigma * C_sqrt @ z
            
            # Evaluate population
            fitnesses = np.zeros(lam)
            for i in range(lam):
                # Clip to box
                x_i = np.clip(samples[i], -5, 5)
                fitnesses[i] = func(x_i)
                calls_made += 1
                
                if fitnesses[i] < best_f:
                    best_f = fitnesses[i]
                    best_x = x_i.copy()
                
                # If we run out of budget mid-population, break
                if calls_made >= self.budget:
                    break
            
            if calls_made >= self.budget:
                break
            
            # Sort by fitness
            idx = np.argsort(fitnesses)
            fitnesses_sorted = fitnesses[idx]
            samples_sorted = samples[idx]
            
            # Update mean
            new_x = np.sum(w[:, np.newaxis] * samples_sorted[:mu], axis=0)
            
            # Update path states
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (new_x - x) / sigma
            norm_ps = np.linalg.norm(ps)
            h_sigma = 1 if norm_ps < (1.4 + 1 / c_sigma) * chi_n else 0
            
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_x - x) / sigma
            
            # Update covariance
            diff = samples_sorted[:mu] - x
            C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - h_sigma) * c_c * (2 - c_c) * C) + \
                c_mu * np.sum(w[:, np.newaxis, np.newaxis] * (diff[:, :, np.newaxis] * diff[:, np.newaxis, :]), axis=0)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
            
            # Update mean
            x = new_x
            
            # Ensure C remains positive definite by adding small jitter if needed
            # Re-orthogonalize C occasionally or just let Cholesky handle it with jitter
            
        return best_x, best_f

# EVOLVE-BLOCK-END
