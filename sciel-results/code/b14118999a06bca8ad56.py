"""CMA-ES with restarts for Rastrigin rotated (f15)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for Rastrigin rotated (f15)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Global best
        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial random sampling to find a decent start
        # Rastrigin has a global min at 0 (before shift/rotation).
        # Let's sample a few points to get a sense of the scale.
        n_init = 10
        init_xs = []
        init_fs = []
        for _ in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = evaluate(x)
            init_xs.append(x)
            init_fs.append(f)
        
        if not init_xs:
            return best_x, best_f

        # Find the best initial point
        init_best_idx = np.argmin(init_fs)
        x0 = init_xs[init_best_idx]
        f0 = init_fs[init_best_idx]
        
        # Estimate sigma based on the spread of good points
        # If points are clustered, sigma should be smaller.
        # Simple heuristic: sigma = 1.0 is a safe start for [-5,5]^5.
        sigma0 = 1.5

        # CMA-ES Parameters
        lam = 10  # Population size
        mu = lam // 2  # Number of parents
        
        # Weights
        w_raw = np.log(np.arange(1, mu + 1))
        w_raw /= w_raw.sum()
        mu_w = (w_raw ** 2).sum()
        mu_eff = 1.0 / mu_w

        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 1) / (mu_eff + 2) ** 2 - 1))
        chi_n = np.sqrt(dim * (1 - 1 / (4 * dim) - 1 / (21 * dim ** 2)))

        # Restart logic
        max_generations_per_run = 50
        stagnation_threshold = 1e-6
        stagnation_counter = 0
        best_f_prev = f0
        restart_count = 0
        max_restarts = 3

        while calls < budget and time.time() < deadline:
            # Initialize CMA-ES state
            mean = x0.copy()
            sigma = sigma0
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            
            # Precompute C_sqrt
            # C = B @ diag(D^2) @ B.T
            C_sqrt = (B * D) @ B.T
            C_sqrt_inv = (B / D) @ B.T
            
            gen = 0
            local_best_f = f0
            local_best_x = x0.copy()

            while calls < budget and time.time() < deadline and gen < max_generations_per_run:
                gen += 1
                # Generate samples
                Z = rng.standard_normal((lam, dim))
                Y = Z @ C_sqrt
                X = mean[None, :] + sigma * Y
                X = np.clip(X, -5.0, 5.0)

                # Evaluate
                F = np.empty(lam)
                for i in range(lam):
                    if calls >= budget or time.time() > deadline:
                        F[i:] = np.inf
                        break
                    F[i] = evaluate(X[i])
                
                # Handle if we broke out of evaluation loop early
                if calls >= budget or time.time() > deadline:
                    break

                # Update best
                idx_min = np.argmin(F)
                if F[idx_min] < local_best_f:
                    local_best_f = F[idx_min]
                    local_best_x = X[idx_min].copy()
                
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()

                # Sort
                order = np.argsort(F)
                X_sorted = X[order]
                F_sorted = F[order]

                # Update mean
                mean_old = mean.copy()
                mean = mean + sigma * np.sum(w_raw[:, None] * X_sorted[:mu], axis=0)

                # Update step size (CSA)
                y = (mean - mean_old) / sigma
                ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_w) / chi_n * C_sqrt_inv @ y
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1))) / chi_n < (1.4 + 2 / (dim + 1)) * c_sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Update covariance matrix
                delta_sig = 1 if hsig else 0
                pc = (1 - c_c) * pc + c_c * delta_sig * (y / sigma)
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    zi = C_sqrt_inv @ (X_sorted[i] - mean_old) / sigma
                    rank_mu += w_raw[i] * np.outer(zi, zi)
                
                C = (1 + c_c * (2 - delta_sig - c_c) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Ensure C is symmetric
                C = (C + C.T) / 2.0

                # Eigendecomposition to update B and D
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12) # Numerical stability
                D = np.sqrt(eigvals)
                B = eigvecs
                C_sqrt = (B * D) @ B.T
                C_sqrt_inv = (B / D) @ B.T

                # Stagnation check
                if local_best_f < best_f_prev - stagnation_threshold:
                    best_f_prev = local_best_f
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                if stagnation_counter > 20:
                    break

            # If we broke out due to stagnation or max gens, consider restarting
            if time.time() > deadline or calls >= budget:
                break

            if stagnation_counter > 20 and restart_count < max_restarts:
                restart_count += 1
                # Restart from the best point found so far with a larger sigma to escape local min
                x0 = best_x.copy()
                sigma0 = max(sigma * 2.0, 1.0)
                # Optionally, if best_f is very poor, restart from random
                if best_f > 50: 
                    x0 = rng.uniform(-5, 5, dim)
                    f0 = evaluate(x0)
                    if f0 < best_f:
                        best_f = f0
                        best_x = x0.copy()
                    sigma0 = 1.5
            else:
                # If no restart or max restarts reached, break
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
