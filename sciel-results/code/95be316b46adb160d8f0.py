"""Fresh design: CMA-ES with restarts + time deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES (with adaptive lambda & restarts) on [-5,5]^dim, 500 calls."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin well under 300s
        calls = 0

        def F(x):
            nonlocal calls
            calls += 1
            return func(x)

        rng = np.random.default_rng(12345)

        # Best-so-far global tracker
        best_x = rng.uniform(-5, 5, dim)
        best_f = F(best_x)

        # ---- CMA-ES single run helper ----
        def cma_run(x0, f0, budget_left):
            nonlocal best_x, best_f, calls
            n = dim
            # population size
            m = int(4 + 3 * np.log(n))
            # step size: start moderate
            sigma = 1.0
            # covariance
            C = np.eye(n)
            # evolution path
            ps = np.zeros(n)
            pc = np.zeros(n)
            # adaptation rates
            mu = m // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / w.sum()
            c_sigma = (mueff + 2) / ((n + 2) * (mueff + 2))
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / ((n + 4) * (4 + mueff / n))
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            # inverse sqrt of C via Cholesky
            def invsqrt(C):
                L = np.linalg.cholesky(C + 1e-20 * np.eye(n))
                # solve L y = e, etc. Use eig for stability
                ev, V = np.linalg.eigh(C)
                ev = np.maximum(ev, 1e-20)
                return V @ np.diag(1.0 / np.sqrt(ev)) @ V.T

            B = None
            D = None
            invsqrtC = invsqrt(C)

            def update_ds():
                nonlocal B, D, invsqrtC
                ev, V = np.linalg.eigh(C)
                ev = np.maximum(ev, 1e-20)
                D = np.sqrt(ev)
                B = V
                invsqrtC = V @ np.diag(1.0 / np.sqrt(ev)) @ V.T

            update_ds()

            x = x0.copy()
            fx = f0
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            while calls < budget_left and (time.time() < deadline - 5.0):
                # sample offspring
                idx = np.arange(m)
                # sample standard normals
                z = rng.standard_normal((m, n))
                # transform: x + sigma * B D z
                off = z @ (B * D).T
                off *= sigma
                y = x[None, :] + off
                y = np.clip(y, -5, 5)
                # evaluate
                fy = np.empty(m)
                for i in range(m):
                    if calls >= budget_left or time.time() >= deadline - 5.0:
                        break
                    fy[i] = F(y[i])
                # if we ran out, break
                if calls >= budget_left:
                    break
                # sort
                order = np.argsort(fy)
                # update best
                if fy[order[0]] < best_f:
                    best_f = fy[order[0]]
                    best_x = y[order[0]].copy()
                # recompute mean of best mu
                wsum = w.sum()
                mean_new = np.sum(w[:, None] * y[order[:mu]], axis=0) / wsum
                # evolution path sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (invsqrtC @ (mean_new - x) / sigma)
                # sigma update
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = min(sigma, 1e6)
                sigma = max(sigma, 1e-12)
                # covariance update
                yw = y[order[:mu]]
                diff = (yw - x) / sigma
                delta = (mean_new - x) / sigma
                pc = (1 - c_c) * pc + c_c * delta
                outer = np.outer(pc, pc)
                C = (1 + c_1 + 2 * c_mu - c_c) * C + c_1 * (outer - C) \
                    + c_mu * np.sum(w[:, None, None] * diff[:, None, :] * diff[:, None, :], axis=0)
                # recompute
                update_ds()
                # accept mean
                x = mean_new
                fx = fy[order[0]]

                # termination: convergence
                if sigma < 1e-12 or np.all(np.abs(C) < 1e-14):
                    break

            return x, fx

        # ---- Multi-start / restart loop ----
        # First a broad random scan to find a good start, then CMA-ES.
        # Use a fraction of budget for initial scan.
        scan_budget = min(20, budget // 10)
        for _ in range(scan_budget):
            if calls >= budget or time.time() >= deadline - 5.0:
                break
            x_try = rng.uniform(-5, 5, dim)
            f_try = F(x_try)
            if f_try < best_f:
                best_f, best_x = f_try, x_try.copy()

        # Run CMA-ES from the best point, with restarts if it stalls.
        # We'll run CMA-ES in phases, restarting from the best point each time
        # but with a fresh (identity) covariance and a randomized sigma.
        restarts = 3
        phase_budget = (budget - calls) // max(1, restarts)
        for r in range(restarts):
            if calls >= budget or time.time() >= deadline - 5.0:
                break
            # fresh covariance each restart, sigma scaled by distance
            # Use best_x as start; sigma based on spread
            sigma0 = 0.5 if r == 0 else 1.0
            # We re-run cma_run but it uses its own sigma=1.0; instead pass via x0 only.
            # To vary sigma, we temporarily set a module-level start sigma:
            cma_start_sigma = sigma0
            # Run with remaining budget
            x_end, f_end = cma_run(best_x.copy(), best_f, budget)
        return best_x, best_f
# EVOLVE-BLOCK-END
