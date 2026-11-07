# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """CMA-ES optimizer for Katsuura function f23 with random restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def cma_es(x0, f0, max_evals, sigma0=None):
            nonlocal best_x, best_f, evals_used
            if sigma0 is None:
                sigma0 = 1.0

            n = dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = max(1, lam // 2)  # number of selected

            # CMA-ES parameters
            c_sigma = (mu + 2) / (n + mu + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + n / mu) / (n + 4 + 2 * mu / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu)
            c_mu = min(1 - c_1, 2 * ((mu + 1) / (n + 2) ** 2 - 1))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # State
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            evals = 0

            # Precompute weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            nu_eff = 1 / np.sum(weights ** 2)

            # Cholesky factor
            try:
                C_sqrt = np.linalg.cholesky(C)
            except:
                C_sqrt = np.eye(n)

            for gen in range(max_evals // lam + 1):
                if time.time() > deadline:
                    break
                if evals_used >= budget:
                    break
                if evals >= max_evals:
                    break

                # Generate population
                offspring = np.zeros((lam, n))
                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = C_sqrt @ z
                    offspring[i] = x_mean + sigma * y

                # Evaluate offspring
                offspring_f = np.zeros(lam)
                for i in range(lam):
                    if time.time() > deadline:
                        break
                    if evals_used >= budget:
                        break
                    x_trial = clip(offspring[i])
                    f_trial = func(x_trial)
                    offspring_f[i] = f_trial
                    evals_used += 1
                    evals += 1

                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = x_trial.copy()

                # Sort offspring by fitness
                idx = np.argsort(offspring_f)
                y_sorted = offspring[idx]
                f_sorted = offspring_f[idx]

                # Update mean
                x_mean_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
                x_mean = x_mean_new

                # Update step size
                y_sigma = (C_sqrt @ (np.sum(weights[:, None] * y_sorted[:mu], axis=0) / sigma))
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * nu_eff) * y_sigma
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (evals / (lam * d_sigma)))) < (1.4 + 2 / (n + 1)) * chi_n
                if h_sigma:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update evolution path for covariance
                y_c = (C @ (np.sum(weights[:, None] * y_sorted[:mu], axis=0) / sigma))
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * nu_eff) * y_c

                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.sum(weights[:, None, None] * np.outer(y_sorted[:mu], y_sorted[:mu]), axis=0) / (sigma ** 2)
                C = (1 + c_1 * (delta_h - 1) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Ensure positive definiteness
                try:
                    C_sqrt = np.linalg.cholesky(C)
                    C_inv = np.linalg.inv(C)
                except:
                    # Regularize
                    C += 1e-6 * np.eye(n)
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                        C_inv = np.linalg.inv(C)
                    except:
                        C = np.eye(n)
                        C_sqrt = np.eye(n)
                        C_inv = np.eye(n)

                # Termination criteria
                if sigma < 1e-10:
                    break
                if np.max(C) / np.min(C) > 1e14:
                    break

            return x_mean, f0

        # Strategy:
        # Phase 1: Quick random sampling to find good starting points
        n_random = 30
        evals_used = 0

        random_points = []
        for i in range(n_random):
            if time.time() > deadline or evals_used >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            evals_used += 1
            random_points.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Sort by fitness
        random_points.sort(key=lambda t: t[0])

        if evals_used >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Run CMA-ES from the best few starting points
        remaining_budget = budget - evals_used
        n_restarts = 3
        evals_per_restart = remaining_budget // n_restarts

        for i in range(min(n_restarts, len(random_points))):
            if time.time() > deadline or evals_used >= budget:
                break

            f_init, x_init = random_points[i]
            sigma0 = 0.5  # Start with smaller sigma since we're already in a good region

            cma_es(x_init, f_init, evals_per_restart, sigma0)

        return best_x, best_f
# EVOLVE-BLOCK-END
