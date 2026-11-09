"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls = 0
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin before 300s limit

        def count_call(x):
            nonlocal calls
            if calls >= self.budget:
                raise RuntimeError("Budget exceeded")
            calls += 1
            return func(x)

        # CMA-ES parameters
        dim = self.dim
        pop_size = 4 + int(3 * np.log(dim))
        lambda_pop = pop_size
        mu = lambda_pop // 2
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def cma_es(x0, f0, budget_remaining):
            nonlocal best_x, best_f, calls
            if budget_remaining <= 0:
                return

            # Initialize CMA-ES state
            x_mean = x0.copy()
            sigma = 1.0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            rootC = np.eye(dim)
            invrootC = np.eye(dim)

            # Precompute weights
            w = weights
            mu_eff = 1.0 / np.sum(w ** 2)

            # Initial population
            for _ in range(budget_remaining):
                if time.time() > deadline:
                    break
                if calls >= self.budget:
                    break

                # Generate population
                ys = []
                xs = []
                for i in range(lambda_pop):
                    if calls >= self.budget or time.time() > deadline:
                        break
                    y = rng.standard_normal(dim)
                    x_new = np.clip(x_mean + sigma * (rootC @ y), -5, 5)
                    fy = count_call(x_new)
                    ys.append(y)
                    xs.append((x_new, fy))

                if len(xs) < 2:
                    break

                # Sort by fitness
                xs.sort(key=lambda t: t[1])
                selected = xs[:mu]

                # Update best
                if selected[0][1] < best_f:
                    best_f = selected[0][1]
                    best_x = selected[0][0].copy()

                # Compute new mean
                x_new_mean = np.sum([w[i] * selected[i][0] for i in range(mu)], axis=0)

                # Update covariance matrix
                if mu_eff > 1e-10:
                    # Damping factor
                    damps = 1 - np.exp(-2 * (calls / (dim + 2)) / (dim * lambda_pop))
                    hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * calls))) < \
                           (1 + 2 / (dim + 1)) * chi_n

                    # Update step size
                    c_sigma_update = c_sigma * (np.linalg.norm(ps) / chi_n - 1)
                    sigma *= np.exp(c_sigma_update)

                    # Update evolution path
                    ps = (1 - c_sigma) * ps + c_sigma * (hsig * 1.0) * (
                        np.linalg.inv(rootC) @ (x_new_mean - x_mean) / sigma
                    )

                    # Update covariance
                    delta_d = c_c * (1 - c_c) if hsig else 0
                    rank_one = np.outer(x_new_mean - x_mean, x_new_mean - x_mean)
                    rank_mu = np.sum([
                        w[i] * np.outer(
                            (rootC @ (selected[i][0] - x_mean) / sigma),
                            (rootC @ (selected[i][0] - x_mean) / sigma)
                        )
                        for i in range(mu)
                    ], axis=0)

                    C = (1 + c_1 * (2 - c_1) + c_mu - c_1) * C \
                        - c_1 * C @ C \
                        + (c_1 + delta_d) * np.outer(x_new_mean - x_mean, x_new_mean - x_mean) / sigma ** 2 \
                        + c_mu * rank_mu

                    # Ensure positive semi-definite
                    C = (C + C.T) / 2
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-10)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                    # Update square root
                    rootC = eigvecs @ np.sqrt(np.diag(eigvals))
                    invrootC = np.linalg.inv(rootC)

                x_mean = x_new_mean

                # Check convergence
                if np.std([s[1] for s in xs]) < 1e-10 and np.std([s[0] for s in xs], axis=0).max() < 1e-8:
                    break

            return x_mean, best_f

        # Multi-start with CMA-ES
        num_restarts = 3
        budget_per_restart = self.budget // num_restarts

        for r in range(num_restarts):
            if time.time() > deadline or calls >= self.budget:
                break

            if r == 0:
                # First start: center
                x0 = np.zeros(dim)
            else:
                # Subsequent starts: random points near best
                if best_x is not None:
                    x0 = np.clip(best_x + rng.standard_normal(dim) * 0.5, -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, dim)

            f0 = count_call(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            remaining = min(budget_per_restart, self.budget - calls)
            cma_es(x0, f0, remaining)

        return best_x, best_f
# EVOLVE-BLOCK-END
