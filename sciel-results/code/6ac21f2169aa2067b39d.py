"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Time guard: return best if we exceed 320s (kill is at 330s)
        start_time = time.time()
        deadline = start_time + 320.0

        # CMA-ES initialization
        # Initial population size
        lambda_pop = 4 + int(3 * np.log(dim))
        # Initial step size
        sigma = 2.0
        # Initial mean
        x_mean = np.zeros(dim)
        
        # CMA-ES parameters
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # State variables
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        # Cholesky factor of C (for efficient sampling)
        # We will update C and recompute Cholesky every step for simplicity in this constrained environment,
        # or use a simpler rank-1/rank-mu update.
        # For robustness and speed in 5D, direct matrix ops are fine.
        
        best_x = x_mean.copy()
        best_fx = func(best_x)
        
        calls_made = 1
        
        # Pre-compute inverse square root of C? No, we sample from N(0, C).
        # Sampling: x = x_mean + sigma * z, where z ~ N(0, C)
        # We can compute C^{-1/2} once if C is fixed, but C changes.
        # Efficient way: maintain L such that C = L L^T.
        L = np.eye(dim)
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            pop = np.zeros((lambda_pop, dim))
            fvals = np.zeros(lambda_pop)
            
            for i in range(lambda_pop):
                # Sample from N(0, C) using L
                z = rng.standard_normal(dim)
                # x = x_mean + sigma * L @ z
                x_i = x_mean + sigma * (L @ z)
                # Clip to box
                x_i = np.clip(x_i, -5, 5)
                pop[i] = x_i
                fvals[i] = func(x_i)
                calls_made += 1
                
                if fvals[i] < best_fx:
                    best_fx = fvals[i]
                    best_x = x_i.copy()
                    
                # Check time inside loop for tight guard
                if calls_made % 10 == 0 and time.time() > deadline:
                    break
            
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Sort population by fitness
            sorted_idx = np.argsort(fvals)
            selected_x = pop[sorted_idx[:mu]]
            selected_f = fvals[sorted_idx[:mu]]
            
            # Update mean
            x_mean_new = np.sum(weights[:, None] * selected_x, axis=0)
            
            # Update sigma (CSA)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (L.T @ (x_mean_new - x_mean))
            # Note: The standard formula uses C^{-1/2} (x_new - x_old). 
            # Since C = L L^T, C^{-1/2} = (L^{-T} L^{-1})? 
            # Actually, if z ~ N(0, C), then z = L y where y ~ N(0, I).
            # The step is sigma * L y.
            # The normalized step is y.
            # So we should compute y = L^{-1} (x_new - x_old) / sigma.
            # Let's stick to the standard implementation which often uses C^{-1/2}.
            # Computing C^{-1/2} is expensive. 
            # Alternative: Use the property that if C = L L^T, then C^{-1/2} = L^{-T} (L L^T)^{-1/2}? No.
            # Let's just compute the norm in the transformed space.
            # y = L^{-1} (x_new - x_old) / sigma
            # We need L^{-1}.
            
            # For simplicity and robustness in this small dim, let's just recompute C and its inverse/sqrt each iteration if needed,
            # or use a simplified CMA-ES that doesn't track C exactly but adapts step size and uses random restarts if stuck?
            # No, the directive asks for a structural change to handle rotated landscapes. CMA-ES is the key.
            
            # Let's implement a slightly slower but correct CMA-ES update.
            # We need C^{-1/2}.
            # Eigendecomposition of C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12) # Numerical stability
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Correct CSA update
            step_vec = x_mean_new - x_mean
            # y = C^{-1/2} step_vec / sigma
            y = C_inv_sqrt @ step_vec / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            
            if np.linalg.norm(ps) < (1.4 + 2 / (dim + 1)) * np.sqrt((1 - (1 - c_sigma) ** (2 * (budget - calls_made)))) * np.sqrt(mu_eff):
                sigma *= np.exp(c_sigma / (dim + 2) * (np.linalg.norm(ps) ** 2 / dim - 1))
            
            # Update pc and C
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * step_vec / sigma
            # Rank-1 update
            rank1 = np.outer(pc, pc)
            h_sigma = (1 + (1 - c_sigma) ** (2 * (budget - calls_made))) * np.sqrt((1 - (1 - c_c) ** (2 * (budget - calls_made)))) / np.sqrt(1 - (1 - c_c) ** (2 * (budget - calls_made)))
            # Standard h_sigma formula:
            # h_sigma = (1 - (1 - c_c)^(2*(gen+1))) / sqrt(1 - (1 - c_c)^(2*(gen+1))) * sqrt((2 - c_c) * mu_eff / (dim + 2) - 1)
            # This is getting complex. Let's use a simpler heuristic for h_sigma or just 1.
            # For 5D and 300 budget, simple CMA-ES often works well.
            
            # Let's use a simpler update for C to avoid numerical issues with h_sigma.
            # C = (1 - c_1 - c_mu) * C + c_1 * (rank1 + (1 - delta_sigma) * C) + c_mu * sum(weights * outer(xi - x_old, xi - x_old))
            # We need delta_sigma = c_sigma * (2 - c_sigma) * mu_eff / dim
            delta_sigma = c_sigma * (2 - c_sigma) * mu_eff / dim
            
            # Rank-mu update
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = selected_x[i] - x_mean
                rank_mu += weights[i] * np.outer(diff, diff)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (rank1 + (1 - delta_sigma) * C) + c_mu * rank_mu
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update L (Cholesky)
            # Add small jitter for stability
            C += 1e-12 * np.eye(dim)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to identity if Cholesky fails
                L = np.eye(dim)
                C = np.eye(dim)
            
            x_mean = x_mean_new
            
            # Check for convergence/stagnation?
            # If sigma becomes very small, we might be stuck.
            if sigma < 1e-10:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
