"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population and restarts for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_run(self, func, x0, sigma0, budget, deadline, rng):
        """Run CMA-ES from x0 with sigma0 for up to budget evaluations or until deadline."""
        n = self.dim
        calls = 0
        best_x = x0.copy()
        best_f = func(x0)
        calls += 1
        if calls >= budget or time.time() >= deadline:
            return best_x, best_f, calls

        # CMA-ES parameters
        popsize = max(4, int(4 + 3 * np.log(n)))
        c_sigma = (popsize + 2) / ((n + 1.3) ** 2 + popsize)
        d_sigma = 1 + 2 * max(0, np.sqrt((popsize - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + popsize / n) / (4 + n + popsize / n)
        c_1 = 2 / ((n + 1.3) ** 2 + popsize)
        c_mu = min(1 - c_1, 2 * ((popsize / 2 + 1) - n - 2) / ((n + 2) ** 2 + popsize))

        xmean = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        cs = (c_c + 1) / (n + c_c)
        ds = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 1)) - 1) + c_c
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        for iteration in range(budget - calls):
            if time.time() >= deadline:
                break
            if calls >= budget:
                break

            # Generate population
            B = np.linalg.cholesky(C)
            D = np.sqrt(np.diag(B @ B.T))
            B_inv = np.linalg.inv(B)

            ys = []
            ws = []
            for i in range(popsize):
                y = rng.standard_normal(n)
                y = B @ y
                x = np.clip(xmean + sigma * y, -5, 5)
                f = func(x)
                calls += 1
                ys.append(y)
                ws.append(f)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                if calls >= budget:
                    break

            if len(ys) < popsize:
                popsize_actual = len(ys)
                ys = ys[:popsize_actual]
                ws = ws[:popsize_actual]
            else:
                popsize_actual = popsize

            if popsize_actual < 2:
                break

            # Sort by fitness
            idx = np.argsort(ws)
            ys_sorted = np.array(ys[idx])
            ws_sorted = np.array(ws[idx])

            # Weights
            mu_eff = popsize_actual // 2
            weights = np.zeros(popsize_actual)
            weights[:mu_eff] = np.log(mu_eff + 0.5) - np.log(np.arange(1, mu_eff + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Update mean
            xmean = np.sum(weights[:, None] * ys_sorted[:mu_eff], axis=0) * sigma + xmean

            # Update step size
            ymean = np.sum(weights[:, None] * ys_sorted[:mu_eff], axis=0)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * B_inv @ ymean
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update covariance
            pc = (1 - c_c) * pc + c_c * ymean
            v = (c_1 + c_mu) * (1 - c_1) + c_mu * (np.sum(weights[:mu_eff] ** 2) - 1)
            C = (1 + c_1 + c_mu - c_1 * (n + 1) + c_mu * (np.sum(weights[:mu_eff] ** 2) - 1)) * C \
                + c_1 * np.outer(pc, pc) \
                + c_mu * np.sum(weights[:mu_eff, None] * np.outer(ys_sorted[:mu_eff], ys_sorted[:mu_eff]), axis=0)

            # Ensure C is positive definite
            C = (C + C.T) / 2
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += (1e-10 - np.min(eigvals)) * np.eye(n)

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10 second safety margin
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Phase 1: Try a few different starting points with short CMA runs
        # Then use remaining budget for one longer CMA run from the best point

        # Initial exploration: evaluate a few random points
        n_init = min(10, budget // 10)
        init_points = []
        init_fvals = []
        for i in range(n_init):
            if time.time() >= deadline or calls_used >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_used += 1
            init_points.append(x)
            init_fvals.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: Run CMA-ES from the best initial point with remaining budget
        remaining_budget = budget - calls_used
        if remaining_budget > 10:
            x0 = best_x.copy() if best_x is not None else rng.uniform(-5, 5, dim)
            sigma0 = 2.0

            # Run CMA with a portion of remaining budget
            cma_budget = remaining_budget
            x_opt, f_opt, calls_cma = self._cma_run(
                func, x0, sigma0, cma_budget, deadline, rng
            )
            calls_used += calls_cma
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # Phase 3: If we have budget left, do local refinement
        if calls_used < budget and best_x is not None:
            # Local search around best point
            local_budget = budget - calls_used
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(local_budget):
                if time.time() >= deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
