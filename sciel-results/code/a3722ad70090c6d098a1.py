"""Rewritten optimizer using CMA-ES with adaptive step-size and restarts."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES based black-box continuous optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(d))
        mu = lambda_ // 2
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = int(1 / np.sum(weights ** 2))
        # Step size initialization
        sigma0 = 2.0
        # Learning rates
        c_sigma = (mueff + 2) / (d + mueff + 5)
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff - 2 + 1 / mueff) / (d + 2) ** 2 + mueff))
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        best_x = None
        best_f = np.inf
        calls_used = 0

        def eval_func(x):
            nonlocal calls_used
            calls_used += 1
            return func(x)

        while calls_used < budget:
            # Initialize population around best or random start
            if best_x is None:
                x0 = rng.uniform(-4, 4, d)
            else:
                x0 = best_x.copy()

            sigma = sigma0
            # Covariance matrix
            C = np.eye(d)
            # Evolution path for covariance
            pc = np.zeros(d)
            # Evolution path for sigma
            ps = np.zeros(d)
            # Mean
            mean = x0.copy()

            # Evaluate initial mean
            f_mean = eval_func(mean)
            if f_mean < best_f:
                best_f = f_mean
                best_x = mean.copy()

            # Track history for stopping
            best_f_hist = []
            stall_counter = 0
            last_best_f = best_f
            no_improve_steps = 0
            max_no_improve = 200  # restart if no improvement for this many steps

            while calls_used + lambda_ <= budget:
                # Generate population
                # Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(d)

                pop = np.empty((lambda_, d))
                for i in range(lambda_):
                    z = rng.standard_normal(d)
                    pop[i] = mean + sigma * (L @ z)
                    pop[i] = np.clip(pop[i], -5, 5)

                # Evaluate population
                f_vals = np.array([eval_func(pop[i]) for i in range(lambda_)])

                # Sort
                idx = np.argsort(f_vals)
                y_sorted = pop[idx]
                f_sorted = f_vals[idx]

                # Update best
                if f_sorted[0] < best_f:
                    best_f = f_sorted[0]
                    best_x = y_sorted[0].copy()
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1

                # Update mean
                w = weights
                new_mean = np.sum(w[:, None] * y_sorted[:mu], axis=0)

                # Update evolution paths
                y_mean = (new_mean - mean) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_mean
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (calls_used // lambda_ + 1))) <= \
                       (1.4 + 2 / (d + 1)) * chi_n
                if hsig:
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * y_mean
                else:
                    pc = (1 - c_c) * pc

                # Update covariance
                delta = 1.0 if not hsig else 0.0
                C = (1 + c_1 * delta - c_1 - c_mu) * C + \
                    c_1 * (np.outer(pc, pc) - C) + \
                    c_mu * np.sum(w[:, None, None] * np.outer(y_sorted[:mu] - mean, y_sorted[:mu] - mean), axis=0)

                # Update sigma
                sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                mean = new_mean

                # Stopping criteria
                if no_improve_steps > max_no_improve:
                    break
                if sigma < 1e-8:
                    break
                if calls_used + lambda_ > budget:
                    break

            # If we still have budget left and didn't use all, do local search around best
            if calls_used < budget:
                # Simple local refinement
                local_sigma = 0.1
                for _ in range(budget - calls_used):
                    if calls_used >= budget:
                        break
                    step = rng.standard_normal(d) * local_sigma
                    candidate = np.clip(best_x + step, -5, 5)
                    f_cand = eval_func(candidate)
                    if f_cand < best_f:
                        best_f = f_cand
                        best_x = candidate.copy()
                        local_sigma *= 1.2
                    else:
                        local_sigma *= 0.9
                    if local_sigma < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
