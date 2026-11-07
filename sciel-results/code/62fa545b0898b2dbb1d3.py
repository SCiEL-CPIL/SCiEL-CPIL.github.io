"""Optimizer for BBOB-style f15 (Rastrigin rotated) with CMA-ES and multi-restart."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multiple restarts for rotated Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        def cma_es(start_x, start_sigma, max_calls, deadline):
            """Run CMA-ES from start_x with given budget."""
            nonlocal calls_made, best_x, best_f, deadline

            dim = self.dim
            pop_size = max(4, int(4 + 3 * np.log(dim)))  # ~12 for dim=5
            mean = start_x.copy()
            sigma = start_sigma
            C = np.eye(dim)
            C_inv = np.eye(dim)
            C_sqrt = np.eye(dim)

            # Update rates
            mu = pop_size // 2
            weights = np.log(np.arange(mu, 0, -1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            def update_C_sqrt(C, n_iter):
                """Approximate Cholesky-like update for C."""
                # Use eigendecomposition for stability
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                return eigvecs @ np.diag(np.sqrt(eigvals))

            C_sqrt = update_C_sqrt(C, 0)

            for iteration in range(max_calls):
                if time.time() > deadline or calls_made >= budget:
                    break

                # Generate population
                y_list = []
                x_list = []
                f_list = []

                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    y = C_sqrt @ z
                    x = mean + sigma * y
                    x_clipped = np.clip(x, -5, 5)
                    f = func(x_clipped)
                    calls_made += 1
                    y_list.append(y)
                    x_list.append(x_clipped)
                    f_list.append(f)

                    if f < best_f:
                        best_f = f
                        best_x = x_clipped.copy()

                # Sort by fitness
                order = np.argsort(f_list)
                x_sorted = [x_list[i] for i in order]
                f_sorted = [f_list[i] for i in order]
                y_sorted = [y_list[i] for i in order]

                # Update mean (weighted recombination)
                old_mean = mean.copy()
                mean = sum(weights[i] * x_sorted[i] for i in range(mu))

                # Update step size (CSA)
                h_sigma = min(1.0,
                              np.sqrt((1 - (1 - c_sigma) ** (iteration + 1)) /
                                      (1 - (1 - c_sigma) ** (2 * (iteration + 1)))) *
                              chi_n * np.linalg.norm(C_sqrt @ (mean - old_mean) / sigma) /
                              np.linalg.norm(C_sqrt @ old_mean + 1e-16)
                              )
                h_sigma = 1.0 if np.linalg.norm(mean - old_mean) < 1e-16 else h_sigma
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * h_sigma * (mean - old_mean) / sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update covariance matrix
                y_diff = [(x_sorted[i] - old_mean) / sigma for i in range(mu)]
                rank_mu = np.outer(y_diff[0], y_diff[0])
                for i in range(1, mu):
                    rank_mu += weights[i] * np.outer(y_diff[i], y_diff[i])

                h_c = min(1.0,
                          np.sqrt((1 - (1 - c_c) ** (iteration + 1)) /
                                  (1 - (1 - c_c) ** (2 * (iteration + 1)))) *
                          chi_n * np.linalg.norm(p_c) / chi_n
                          )
                h_c = 1.0 if np.linalg.norm(p_c) < 1e-16 else h_c
                p_c = (1 - c_c) * p_c + c_c * h_c * (mean - old_mean) / sigma

                C = (1 + c_1 + c_mu - c_1 - c_mu / dim) * C \
                    + c_1 * (np.outer(p_c, p_c) + (1 - h_c) * c_c * (2 - c_c) * C) \
                    + c_mu * rank_mu

                # Ensure positive definiteness
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                C_sqrt = update_C_sqrt(C, iteration)

            return best_x, best_f

        # Phase 1: Initial diverse evaluation
        init_points = [np.zeros(dim)]
        for v in [0.5, -0.5, 1.0, -1.0, 2.0, -2.0]:
            init_points.append(np.full(dim, v))
        for i in range(dim):
            p = np.zeros(dim)
            p[i] = 1.0
            init_points.append(p.copy())
            p[i] = -1.0
            init_points.append(p.copy())
        for _ in range(10):
            init_points.append(rng.uniform(-4, 4, dim))

        for ip in init_points:
            if calls_made >= budget or time.time() > deadline:
                break
            evaluate(ip)

        # Phase 2: Multi-restart CMA-ES with decreasing budgets
        if best_x is not None and calls_made < budget and time.time() < deadline:
            # Allocate remaining budget across restarts
            remaining = budget - calls_made
            n_restarts = min(5, max(1, remaining // 50))
            budgets = []
            base = remaining // n_restarts
            for i in range(n_restarts):
                if i < n_restarts - 1:
                    budgets.append(base)
                else:
                    budgets.append(remaining - sum(budgets))

            for r in range(n_restarts):
                if calls_made >= budget or time.time() > deadline:
                    break

                if r == 0:
                    start_x = best_x.copy()
                    start_sigma = 1.5
                else:
                    # New random start, biased toward best region
                    center = best_x + rng.uniform(-3, 3, dim)
                    center = np.clip(center, -4.5, 4.5)
                    f_new = func(center)
                    calls_made += 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = center.copy()
                    start_x = center.copy()
                    start_sigma = 1.0

                max_calls = budgets[r]
                if max_calls < 10:
                    continue

                cma_es(start_x, start_sigma, max_calls, deadline)

        # Phase 3: Final polish with small CMA-ES
        if best_x is not None and calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining >= 10:
                cma_es(best_x.copy(), 0.3, remaining, deadline)

        # Fallback
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
