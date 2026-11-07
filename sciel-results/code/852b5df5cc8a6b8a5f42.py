"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Pure CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.perf_counter()
        deadline = start + 285.0  # Safety margin well under the 300s/330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # CMA-ES Parameters
        # For dim=5, lambda=10 is a robust choice (4 + floor(3*ln(5)) = 7, but 10 is often better for small dims)
        lambda_pop = 10
        mu = lambda_pop // 2  # Number of best individuals used for update
        
        # Weights
        # Standard weights: w_i = ln(mu/2 + 1) - ln(i) for i=1..mu, normalized
        w = np.array([np.log(mu / 2 + 1) - np.log(i) for i in range(1, mu + 1)])
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        
        # Step size initialization
        # Start with a step size that covers a significant portion of the box [-5, 5]
        # Box width is 10. A sigma of 2.0 or 2.5 is reasonable for initial exploration.
        sigma = 2.0
        
        # Means of weights for update rates
        c_sigma = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_c = (4 + mu_eff / dim) / (dim + 9 + 3 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff / 2 - 1) / ((dim + 2) ** 2 + mu_eff))
        
        # Damping
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        
        # Initial state
        # Random start in the center of the box
        x = rng.uniform(-1, 1, dim)
        fx = func(x)
        calls = 1
        
        best_x = x.copy()
        best_f = fx
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Inverse square root of C for sampling and updates
        # We need to compute C^{-1/2}. For small dim, we can use eigendecomposition.
        def update_sqrt_inv(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues (numerical stability)
            eigvals = np.maximum(eigvals, 1e-10)
            inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
            B = eigvecs @ np.diag(inv_sqrt_eigvals)
            D = np.diag(np.sqrt(eigvals))
            return B, D, eigvals, eigvecs

        B, D, eigvals, eigvecs = update_sqrt_inv(C)
        
        # Loop until budget or time is exhausted
        while calls < budget:
            if time.perf_counter() > deadline:
                break
            
            # Generate population
            pop_x = []
            pop_f = []
            
            for i in range(lambda_pop):
                if time.perf_counter() > deadline:
                    break
                # Sample: x = mean + sigma * B * z
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ z)
                # Clip to box
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                pop_x.append(y)
                pop_f.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            if len(pop_x) < mu:
                break
                
            # Sort population by fitness
            indices = np.argsort(pop_f)
            pop_x = [pop_x[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]
            
            # Select best mu individuals
            x_best = np.array(pop_x[:mu])
            
            # New mean
            x_new = np.sum(w[:, None] * x_best, axis=0)
            
            # Update evolution path p_sigma
            # h_sigma: heuristic for step size adaptation
            # We need to compute ||B^{-1} p_c|| / chi_n
            # B is the matrix such that C = B D B^T. So B^{-1} is B^T D^{-1}.
            # Let's compute B_inv_p_c
            B_inv = np.linalg.inv(B) # dim=5, this is cheap
            norm_p_c_B_inv = np.linalg.norm(B_inv @ p_c)
            h_sigma = 1.0 if (norm_p_c_B_inv / chi_n) < (1.4 + 2 / (dim + 1)) * (1 - c_sigma) else 0.0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B_inv @ (x_new - x) / sigma)
            
            # Update step size
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update evolution path p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            
            # Update covariance matrix
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_sigma) c_c (2 - c_c) C) + c_mu sum w_i (y_i y_i^T)
            # where y_i = (x_i - x) / sigma
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            
            # Precompute sum of outer products of selected individuals
            y_best = (x_best - x) / sigma
            rank_mu = np.sum(w[:, None] * y_best[:, :, None] * y_best[:, None, :], axis=0)
            
            C_new = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + delta_h * C) + c_mu * rank_mu
            
            # Symmetrize
            C_new = (C_new + C_new.T) / 2
            
            # Check eigenvalues for numerical stability
            eigvals_new, _ = np.linalg.eigh(C_new)
            if np.any(eigvals_new < 1e-10):
                # If covariance becomes singular or ill-conditioned, restart or reset
                # For simplicity, we'll just add a small regularization
                C_new += 1e-10 * np.eye(dim)
            
            C = C_new
            x = x_new
            
            # Update B, D
            B, D, eigvals, eigvecs = update_sqrt_inv(C)
            
            # If sigma becomes too small or too large, we might want to restart
            # But for this task, we just run out of budget.
            
            # Optional: If we are stuck (sigma very small and no improvement), we could restart.
            # But CMA-ES usually handles this. Given the short budget, simple CMA-ES is robust.
            
        return best_x, best_f

# EVOLVE-BLOCK-END
