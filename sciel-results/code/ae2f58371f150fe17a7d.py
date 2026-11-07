"""CMA-ES with restarts for Rastrigin-rotated f15."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES with multiple restarts for Rastrigin-rotated f15.
    Uses population-based evolution with covariance adaptation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, budget_remaining, deadline, rng, calls_tracker, best_tracker):
        """Run CMA-ES for a given number of evaluations."""
        dim = self.dim
        n = dim

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = int(np.sum(weights[:int(1.0 / np.sum(weights ** 2))]) ** 2)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = (2 - mueff * c_sigma + c_c) / ((n + 2) ** 1.8 + mueff * c_sigma)
        c_mu = (2 * (mueff * weights[0] - 1) + c_c * c_1) / ((n + 2) ** 1.2 + mueff * weights[0])
        c_mu = min(c_mu, 1 - c_1)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initialize
        x = x0.copy()
        sigma = sigma0
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        step_sigma = np.sqrt(1 - np.exp(-2 / (budget_remaining * 1.0)))

        evals_used = 0
        local_best_f = np.inf
        local_best_x = x.copy()

        for gen in range(10000):
            if calls_tracker[0] + lambda_ > budget_remaining or time.time() > deadline:
                break

            # Sample lambda_ points
            ys = []
            for _ in range(lambda_):
                z = rng.standard_normal(n)
                y = B @ (D * z)
                ys.append(y)
            ys = np.array(ys)

            # Evaluate
            xs = x + sigma * ys
            fs = []
            for i in range(lambda_):
                xi = np.clip(xs[i], -5.0, 5.0)
                fi = func(xi)
                calls_tracker[0] += 1
                fs.append(fi)
                if fi < local_best_f:
                    local_best_f = fi
                    local_best_x = xi.copy()
                if fi < best_tracker[0]:
                    best_tracker[0] = fi
                    best_tracker[1] = xi.copy()

            fs = np.array(fs)
            order = np.argsort(fs)

            # Update mean
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += weights[i] * ys[order[i]]
            x = x + sigma * y_w

            # Update sigma
            y_sigma = np.zeros(n)
            for i in range(mu):
                y_sigma += weights[i] * (ys[order[i]] - y_w)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update covariance
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * y_w
            hs = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
            delta_hs = (1 - hs) if hs else 0
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                zi = ys[order[i]]
                rank_mu += weights[i] * np.outer(zi, zi)
            C = (1 + c_1 * (hs - 1) - c_1) * C + c_1 * rank_one + c_mu * rank_mu
            C = (C + C.T) / 2

            # Eigendecomposition (periodic)
            if gen % 10 == 0:
                try:
                    D2, B2 = np.linalg.eigh(C)
                    D2 = np.sqrt(np.maximum(D2, 1e-20))
                    B = B2
                    D = D2
                except:
                    pass

            # Sigma limit
            sigma = max(sigma, 1e-15)
            sigma = min(sigma, 1e15)

            evals_used += lambda_
            if evals_used >= budget_remaining:
                break

        return local_best_x, local_best_f

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill

        rng = np.random.default_rng(42)
        calls_made = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        calls_tracker = [0]
        best_tracker = [np.inf, np.zeros(dim)]

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            calls_tracker[0] += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            if fx < best_tracker[0]:
                best_tracker[0] = fx
                best_tracker[1] = x.copy()
            return fx

        # Phase 1: Initial exploration with several CMA-ES restarts
        # Start from diverse points
        start_points = [
            np.zeros(dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-3, 3, dim),
        ]

        # Allocate budget across restarts
        # First restart gets more budget for exploration
        n_restarts = 3
        budget_per_restart = budget // (n_restarts + 1)

        for i in range(n_restarts):
            if time.time() > deadline or calls_made >= budget:
                break

            remaining = budget - calls_made
            if remaining < 20:
                break

            sp = start_points[i].copy()
            sigma = 2.0 if i == 0 else 1.0

            # Run CMA-ES
            try:
                local_best_x, local_best_f = self._cmaes(
                    eval_func, sp, sigma, remaining, deadline, rng, calls_tracker, best_tracker
                )
            except Exception:
                pass

            if not time_check():
                break

        # Phase 2: Fine refinement from best found
        if time.time() < deadline and calls_made < budget - 10 and best_f < np.inf:
            # Try a few small-scale CMA-ES runs from the best point
            for sigma in [0.5, 0.2, 0.1, 0.05]:
                if time.time() > deadline or calls_made >= budget:
                    break
                remaining = budget - calls_made
                if remaining < 15:
                    break

                sp = best_x.copy() + 0.1 * rng.standard_normal(dim)
                sp = np.clip(sp, -5, 5)

                try:
                    local_best_x, local_best_f = self._cmaes(
                        eval_func, sp, sigma, remaining, deadline, rng, calls_tracker, best_tracker
                    )
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
