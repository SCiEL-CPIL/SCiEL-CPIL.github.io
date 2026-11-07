"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """CMA-ES based optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def timed_call(x):
            return func(np.asarray(x, dtype=float))

        best_x = None
        best_f = float('inf')
        calls = 0

        def update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        def time_left():
            return deadline - time.time()

        # --- Phase 0: initial sampling ---
        # Sample a few points to get a reasonable starting value
        n_init = 10
        for i in range(n_init):
            if calls >= budget or time_left() <= 5.0:
                break
            x = rng.uniform(-5, 5, dim)
            f = timed_call(x)
            calls += 1
            update(x, f)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float('inf')

        # --- CMA-ES implementation ---
        def cma_es(x0, fx, max_calls):
            nonlocal calls
            x = np.array(x0, dtype=float)
            f = fx
            
            # Population size
            lam = 4 + int(3 * np.log(dim))
            mu = lam // 2
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            # Step size
            sigma = 1.0
            sigma_min = 1e-12
            sigma_max = 1e6
            
            # Covariance matrix and evolution paths
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Parameters
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            
            # Normalization constant
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            # Maximum number of iterations
            max_iter = max_calls // lam + 10
            
            for it in range(max_iter):
                if calls >= budget or time_left() <= 5.0:
                    break
                
                # Generate population
                B, D = np.linalg.eigh(C)
                B = B @ np.diag(D)
                
                z = rng.standard_normal((lam, dim))
                # Normalize z to have unit norm
                norms = np.linalg.norm(z, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                z = z / norms
                z *= chi_n
                
                offspring = x[None, :] + sigma * (z @ B.T)
                offspring = np.clip(offspring, -5, 5)
                
                # Evaluate offspring
                fs = np.empty(lam)
                for i in range(lam):
                    if calls >= budget or time_left() <= 5.0:
                        break
                    fs[i] = timed_call(offspring[i])
                    calls += 1
                    if fs[i] < f:
                        x, f = offspring[i], fs[i]
                        update(x, f)
                
                if calls >= budget or time_left() <= 5.0:
                    break
                
                # Sort offspring
                order = np.argsort(fs)
                x_sorted = offspring[order]
                f_sorted = fs[order]
                
                # Update mean
                x_new = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
                
                # Update evolution paths
                y_vec = (x_new - x) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
                hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) / chi_n
                
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec
                
                # Update covariance matrix
                delta_h = (1.7 + 0.5 / dim) * (1 - (1 - c_c) ** (2 * (it + 1)))
                delta_h *= 1 if hs < (1.4 + 2 / (dim + 1)) * (1 - delta_h) else 0
                
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C
                C += c_1 * (np.outer(pc, pc) - (1 - (1 - c_c) ** 2) * C)
                C += c_mu * np.sum(weights[:mu, None, None] *
                                   np.outer(x_sorted[:mu] - x, x_sorted[:mu] - x) / sigma ** 2,
                                   axis=0)
                
                # Symmetrize and ensure positive definiteness
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-12:
                    C = eigvecs @ np.diag(np.maximum(eigvals, 1e-12)) @ eigvecs.T
                
                # Update step size
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, sigma_min, sigma_max)
                
                x = x_new
                
                # Convergence check
                if np.linalg.norm(y_vec) * sigma < 1e-12:
                    break
            
            return x, f

        # Run CMA-ES
        x_cma, f_cma = cma_es(best_x, best_f, budget - calls)
        update(x_cma, f_cma)

        # Final polish: (1+1)-ES with small sigma if budget remains
        if calls < budget and time_left() > 5.0:
            x = np.array(best_x, dtype=float)
            f = best_f
            sigma = 0.05
            for _ in range(50):
                if calls >= budget or time_left() <= 5.0:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = timed_call(y)
                calls += 1
                if fy < f:
                    x, f = y, fy
                    sigma *= 1.1
                    update(x, f)
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
