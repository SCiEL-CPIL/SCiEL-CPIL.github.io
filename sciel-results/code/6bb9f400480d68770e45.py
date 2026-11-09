"""Optimizer for BBOB f4 (Büche-Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population and restarts for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill

        rng = np.random.default_rng(12345)

        # CMA-ES parameters
        n = self.dim
        default_lambda = max(4 + int(3 * np.log(n)), 10)
        default_sigma = 2.0

        # State
        best_x = None
        best_f = np.inf

        # Budget tracking
        calls_made = 0

        def check_deadline():
            return time.time() >= deadline

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial exploration: a few random points to get a sense of the landscape
        # Use a small number of initial points to avoid wasting budget
        n_init = min(5, self.budget // 10)
        for i in range(n_init):
            if check_deadline():
                break
            x = rng.uniform(-4, 4, n)
            f = evaluate(x)
            if best_x is None:
                best_x = x.copy()

        # Main CMA-ES loop with restarts
        # We'll run CMA-ES until budget is exhausted or deadline
        # Use restarts when the population collapses or progress stalls

        while calls_made < self.budget and not check_deadline():
            # CMA-ES parameters
            lam = default_lambda
            mu = lam // 2
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Initialize CMA-ES state from current best
            x_mean = best_x.copy() if best_x is not None else rng.uniform(-4, 4, n)
            sigma = default_sigma

            # Covariance matrix and evolution paths
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            # Progress tracking for restart
            stall_counter = 0
            best_in_run = best_f
            no_improve_count = 0

            # Run CMA-ES for this restart
            while calls_made < self.budget and not check_deadline():
                # Generate population
                B = rng.standard_normal((lam, n))
                # Apply square root of C
                # Use eigendecomposition for efficiency in low dim
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

                # Generate samples
                pops = x_mean + sigma * (B @ sqrt_C.T)
                pops = np.clip(pops, -5, 5)

                # Evaluate population
                fitness = []
                for i in range(lam):
                    if check_deadline():
                        break
                    f = evaluate(pops[i])
                    fitness.append(f)

                if len(fitness) < lam:
                    # Deadline hit during evaluation
                    break

                fitness = np.array(fitness)
                idx = np.argsort(fitness)

                # Update best
                if fitness[idx[0]] < best_in_run:
                    best_in_run = fitness[idx[0]]
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Update mean
                y_w = np.mean(pops[idx[:mu]], axis=0)
                x_mean_new = x_mean + sigma * np.sum(w[:mu, None] * (pops[idx[:mu]] - x_mean), axis=0) / sigma
                # Actually, the standard update is:
                # x_mean_new = sum(w_i * y_i) where y_i = (pop_i - x_mean) / sigma
                # So: x_mean_new = x_mean + sum(w_i * (pop_i - x_mean))
                x_mean_new = x_mean + np.sum(w[:mu, None] * (pops[idx[:mu]] - x_mean), axis=0)
                x_mean_new = np.clip(x_mean_new, -5, 5)

                # Update evolution paths
                y_sigma = np.sum(w[:mu, None] * (pops[idx[:mu]] - x_mean) / sigma, axis=0)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - lam)))) < (1.4 + 2 / (n + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma

                # Update covariance
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C
                C += c_1 * (np.outer(pc, pc) - C)
                C += c_mu * np.sum(w[:mu, None] * np.outer(
                    (pops[idx[:mu]] - x_mean) / sigma,
                    (pops[idx[:mu]] - x_mean) / sigma
                ), axis=0)

                x_mean = x_mean_new

                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Check for stagnation
                if no_improve_count > 3 * lam:
                    stall_counter += 1
                    if stall_counter >= 2:
                        break
                else:
                    stall_counter = 0

                # Check if sigma is too small
                if sigma < 1e-6:
                    break

            # If we broke out due to stagnation or sigma collapse, restart with new sigma
            # But only if we have budget left
            if calls_made >= self.budget or check_deadline():
                break

            # If we made good progress, we might want to continue with a fresh CMA-ES
            # from the best point found
            # Reset for next restart with a slightly different initial sigma
            default_sigma = max(default_sigma * 0.7, 0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
