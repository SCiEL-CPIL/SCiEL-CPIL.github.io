"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start for Schaffers F7 on [-5, 5]^5.

    CMA-ES adapts a multivariate normal sampling distribution (mean,
    covariance, step size) to the local landscape, making it highly
    effective for continuous, non-linear, non-convex functions.  We run
    several independent CMA-ES restarts: the first uses a large initial
    sigma to explore globally, and subsequent restarts start near the
    current global best with a smaller sigma for exploitation.  A strict
    call budget and wall-clock deadline guard keep the run valid.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 315.0  # safety margin vs 330 s wall-clock kill

        rng = np.random.default_rng(12345)

        # ---- Global bookkeeping ------------------------------------
        x_best_global = None
        f_best_global = np.inf
        n_calls = 0

        def record(x, f):
            nonlocal x_best_global, f_best_global, n_calls
            if f < f_best_global:
                f_best_global = f
                x_best_global = x.copy()
            n_calls += 1

        def check_deadline():
            return n_calls >= budget or time.time() > deadline

        # ---- CMA-ES parameters -------------------------------------
        # Pop size default for CMA-ES: 4 + floor(3 ln(n))
        pop_size = max(4 + int(3 * np.log(dim)), 4)
        # For dim=5: 4 + floor(3*1.609) = 4 + 4 = 8

        # Strategy: multiple restarts
        # Restart 1: broad exploration from center, large sigma
        # Restart 2+: exploit around best with smaller sigma
        # We allocate budget across restarts, giving more to the first.

        restarts = []
        # First restart gets the majority of the budget
        n_restarts = 4
        # Budget split: first restart ~50%, rest share the remainder
        first_budget = int(budget * 0.5)
        remaining = budget - first_budget
        for i in range(n_restarts):
            if i == 0:
                restarts.append(first_budget)
            else:
                restarts.append(remaining // (n_restarts - 1))
        # Ensure the last restart gets any leftover
        restarts[-1] += budget - sum(restarts)

        for restart_idx, restart_budget in enumerate(restarts):
            if check_deadline():
                break

            # Initial mean and sigma
            if restart_idx == 0:
                # Start from center with large sigma for global exploration
                mean = np.zeros(dim)
                sigma = 2.0  # Covers most of [-5,5]^5
            else:
                # Start near the current global best with shrinking sigma
                mean = x_best_global.copy() if x_best_global is not None else np.zeros(dim)
                sigma = max(0.5 / (restart_idx + 1), 0.05)

            # CMA-ES state
            C = np.eye(dim)  # Covariance matrix
            C_inv = np.eye(dim)
            C_sqrt_inv = np.eye(dim)  # Inverse square root of C (for sampling)
            step_size = sigma
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

            # Cumulation paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Adaptation parameters
            c_sigma = (pop_size + 2) / (dim + 4)
            mu = pop_size // 2  # Number of recombination weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (2 - c_c)
            c_mu = (2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff)) * \
                   np.clip(1 - c_1, 0, 0.5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

            # Local budget for this restart
            local_calls = 0
            max_local_calls = min(restart_budget, budget - n_calls)

            while local_calls < max_local_calls and not check_deadline():
                # Generate population
                population = []
                for _ in range(pop_size):
                    # Sample from N(mean, step_size^2 * C)
                    z = rng.standard_normal(dim)
                    # y = C_sqrt_inv @ z, then x = mean + step_size * y
                    # We maintain C_sqrt_inv such that C_sqrt_inv @ C_sqrt_inv^T = C
                    # Actually for sampling: x = mean + step_size * C_sqrt @ z
                    # where C_sqrt @ C_sqrt^T = C
                    # Let's use Cholesky: C = L L^T, sample = L @ z
                    # We'll recompute C_sqrt from C each generation for simplicity
                    # but that's expensive. Instead, we track C_sqrt via updates.
                    # For simplicity and correctness, recompute each gen.
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(dim)
                    y = L @ z
                    x = mean + step_size * y
                    # Clip to bounds
                    x = np.clip(x, -5.0, 5.0)
                    population.append(x)

                # Evaluate population
                fvals = np.array([func(x) for x in population])
                for x, f in zip(population, fvals):
                    record(x, f)
                local_calls += pop_size

                if check_deadline():
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                population = [population[i] for i in order]
                fvals = fvals[order]

                # Weighted mean of the mu best
                x_w = np.zeros(dim)
                for i in range(mu):
                    x_w += w[i] * population[i]

                # Update cumulation paths
                # p_sigma
                if step_size > 1e-14:
                    p_sigma = (1 - c_sigma) * p_sigma + \
                              np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                              (C_sqrt_inv @ (x_w - mean) / step_size)
                # p_c
                h_sigma = 1.0
                if step_size > 1e-14:
                    h_sigma = (np.linalg.norm(p_sigma) /
                               np.sqrt(1 - (1 - c_sigma) ** (2 * (local_calls + 1))) <
                               (1 + 2 / (dim + 1)) * chi_n)

                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * \
                      (C_sqrt_inv @ (x_w - mean) / step_size)

                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_mu_term = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = C_sqrt_inv @ (population[i] - mean) / step_size
                    rank_mu_term += w[i] * np.outer(y_i, y_i)

                C = (1 + c_1 * (delta_h * np.outer(p_c, p_c) - c_1) + c_mu * rank_mu_term) * C

                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Regularize if needed
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    C += 1e-10 * np.eye(dim)

                # Update step size (sigma)
                step_size *= np.exp(c_sigma / d_sigma *
                                    (np.linalg.norm(p_sigma) / chi_n - 1))
                # Clamp step size
                step_size = np.clip(step_size, 1e-8, 10.0)

                # Update mean
                mean = x_w
                mean = np.clip(mean, -5.0, 5.0)

                # Recompute C_sqrt_inv (inverse of Cholesky of C)
                try:
                    L = np.linalg.cholesky(C)
                    C_sqrt_inv = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    C_sqrt_inv = np.eye(dim)

                # Convergence check: if step size is very small, break
                if step_size < 1e-6:
                    break

        return x_best_global, f_best_global
# EVOLVE-BLOCK-END
