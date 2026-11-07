"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin for 300s limit
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        # Initialization
        # Start at a random point to handle hidden shifts
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, n)
        sigma = 2.0
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Covariance matrix adaptation
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Evaluation budget tracking
        calls = 0
        best_x = x.copy()
        best_f = float('inf')
        
        # Initial evaluation
        best_f = func(x)
        calls += 1
        
        # Pre-compute some constants for speed
        sqrt_n = np.sqrt(n)
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            B = rng.standard_normal((lam, n))
            
            # Whitening matrix (inverse sqrt of C)
            # Use eigen decomposition for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            
            # C^{1/2} = V diag(sqrt(lambda)) V^T
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            # C^{-1/2} = V diag(1/sqrt(lambda)) V^T
            inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Sample
            # Y = x + sigma * B @ sqrt_C.T  (Note: B is (lam, n), sqrt_C is (n, n))
            # Actually, standard is Y_i = x + sigma * C^{1/2} z_i
            # If Z is (lam, n), then Y = x + sigma * Z @ C^{1/2}.T ?
            # Let's stick to the vector form logic or matrix multiplication correctly.
            # z_i ~ N(0, I). y_i = x + sigma * C^{1/2} z_i
            # If B is (lam, n) containing z_i, then Y = x + sigma * B @ C^{1/2}.T
            # Since C is symmetric, C^{1/2} is symmetric, so C^{1/2}.T = C^{1/2}
            Y = x + sigma * (B @ sqrt_C.T)
            
            # Clip to box
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.empty(lam)
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    # If we run out of budget mid-generation, break and handle partial generation
                    # For simplicity, if we break, we just exit the loop after this
                    break
                fy = func(Y[i])
                F[i] = fy
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = Y[i].copy()
            
            if calls >= budget or time.time() > deadline:
                break
            
            # If we broke out of the inner loop early, we might have uninitialized F values.
            # Check if all F values are valid
            if not np.all(np.isfinite(F)):
                 # This shouldn't happen if func returns float, but just in case
                 break

            # Sort
            F_sorted_idx = np.argsort(F)
            Y_sorted = Y[F_sorted_idx]
            F_sorted = F[F_sorted_idx]
            
            # Update mean
            x_new = np.sum(w[:, np.newaxis] * Y_sorted[:mu], axis=0)
            
            # Update step size
            y_invC = (Y_sorted[:mu] - x) / sigma
            # ps = (1 - c_sigma) ps + sqrt(c_sigma (2 - c_sigma) mu_eff) * C^{-1/2} * (w^T y)
            # w^T y is a vector of weighted steps in the C^{-1/2} space?
            # Standard formula: ps = (1-c_sigma)ps + sqrt(c_sigma(2-c_sigma)mu_eff) * C^{-1/2} (sum w_i (y_i - x)/sigma)
            # Here y_invC[i] is (y_i - x)/sigma.
            # We need C^{-1/2} @ (sum w_i y_invC[i])
            weighted_sum = np.sum(w[:, np.newaxis] * y_invC, axis=0)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ weighted_sum)
            
            # Update correlation matrix
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            
            # Update covariance
            # h_pc heuristic
            delta_h = (1 - np.exp(-2 + 5 / (n + 1)))
            # Check condition for h_pc
            # norm(ps) / sqrt(1 - (1 - c_sigma)^((gen+1)/d_sigma)) < (1.4 + 2/(n+1)) * trace(C)/n
            # We need generation count. Let's approximate gen as calls // lam
            gen = calls // lam
            if gen > 0:
                denom = np.sqrt(1 - (1 - c_sigma)**((gen + 1) / d_sigma))
            else:
                denom = 1.0
            
            if denom > 1e-12:
                norm_ps = np.linalg.norm(ps)
                threshold = (1.4 + 2 / (n + 1)) * np.trace(C) / n
                if norm_ps / denom < threshold:
                    h_pc = delta_h
                else:
                    h_pc = 0.0
            else:
                h_pc = 0.0
            
            # C = (1 - c_1 - c_mu) C + c_1 (pc pc^T + (1 - h_pc) c_c (2 - c_c) C) + c_mu sum w_i (y_i y_i^T)
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (1 - h_pc) * c_c * (2 - c_c) * C)
            
            for i in range(mu):
                y_i = y_invC[i]
                C += c_mu * w[i] * np.outer(y_i, y_i)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            x = x_new
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / sqrt_n - 1))
            
            # Clip x to reasonable bounds to prevent numerical issues, though func clips
            x = np.clip(x, -10, 10)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
