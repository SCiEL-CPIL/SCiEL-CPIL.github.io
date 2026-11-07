"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 325.0  # 325s safety margin before 330s kill
        rng = np.random.default_rng(0)
        
        # Parameters
        dim = self.dim
        N = max(2, int(4 + 3 * np.log(dim)))  # Population size
        # Initial sigma
        sigma = 2.0
        # Initial mean: uniform in [-5, 5]
        x = rng.uniform(-5, 5, dim)
        
        # CMA-ES state
        C = np.eye(dim)
        # Step size control state
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        # Learning rates
        cs = (np.sqrt(N + 2) - 1) / (N + 4)
        cmu = min(2 / ((N + 4) ** 2 + dim), 2 * cs)
        c1 = (2 + 2 / (N + 1.3)) / ((N + 2) ** 2 + dim)
        c_sigma = cs * (2 + cmu)
        d_sigma = 1 + 2 * max(0, np.sqrt((N - 1) / (dim + 1)) - 1) + cmu
        chi_N = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))
        
        # Evaluate initial point
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        eval_count = 1
        
        # Main loop
        while eval_count < self.budget:
            if time.time() > deadline:
                break
                
            # Generate population
            # Update step size
            # Standard CMA-ES update logic
            
            # 1. Sample population
            # We need to update C and sigma based on previous iteration's results
            # But for the first iteration, we just sample
            
            # Actually, standard CMA-ES loop:
            # 1. Sample y_i
            # 2. x_i = x + sigma * B * D * y_i
            # 3. Evaluate
            # 4. Sort
            # 5. Update x, C, sigma
            
            # Let's do a simplified but robust CMA-ES
            
            # Cholesky factor of C
            L = np.linalg.cholesky(C)
            
            # Sample N points
            Y = rng.standard_normal((N, dim))
            # Transform to x space
            # x_i = x + sigma * L @ Y_i
            X = x[None, :] + sigma * (Y @ L.T)
            
            # Clip to box
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(xi) for xi in X])
            eval_count += N
            
            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_fx:
                best_fx = F[min_idx]
                best_x = X[min_idx].copy()
                
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update mean
            w = np.log(np.arange(1, N + 1)) - np.log(N) / 2
            w = w / w.sum()
            mu = int(np.floor(N / 2))
            w = w[:mu]
            w_sum = w.sum()
            w = w / w_sum
            mu_eff = 1 / np.sum(w ** 2)
            
            x_new = np.sum(X_sorted[:mu] * w[:, None], axis=0)
            
            # Update covariance
            # Delta x
            delta_x = x_new - x
            pc = (1 - cs) * pc + cs * np.sqrt(mu_eff) * delta_x / sigma
            hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cs) ** (2 * (eval_count - N))) < (1.4 + 2 / (dim + 1)) * chi_N)
            hsig = 1 if hsig else 0
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cs * (2 - cs) * C)
            for i in range(mu):
                z = np.linalg.solve(L, (X_sorted[i] - x) / sigma)
                C += cmu * w[i] * np.outer(z, z)
            
            # Update sigma
            ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * (delta_x / sigma)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_N - 1))
            
            # Update mean
            x = x_new
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Eigenvalue check for stability
            eigvals = np.linalg.eigvalsh(C)
            if np.any(eigvals < 1e-12):
                C += 1e-12 * np.eye(dim)

        return best_x, best_fx
# EVOLVE-BLOCK-END
