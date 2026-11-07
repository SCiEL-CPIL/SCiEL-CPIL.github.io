"""Hybrid CMA-ES + multi-start coordinate-descent for Schwefel f20."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for global exploration, coordinate-descent polish for local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls = [0]

        def F(x):
            calls[0] += 1
            return func(x)

        rng = np.random.default_rng(12345)

        # ---- Initial evaluation at center ----
        best_x = np.zeros(dim)
        best_f = F(best_x)

        # ---- Quick random scan for a better starting point ----
        scan_budget = min(12, budget // 20)
        for _ in range(scan_budget):
            if calls[0] >= budget or time.time() >= deadline - 10.0:
                break
            x_try = rng.uniform(-5, 5, dim)
            f_try = F(x_try)
            if f_try < best_f:
                best_f, best_x = f_try, x_try.copy()

        # ---- Coordinate descent polish (exploits Schwefel separability) ----
        def coord_descent_polish(x_start, f_start, call_limit):
            nonlocal best_x, best_f
            x = x_start.copy()
            fx = f_start
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            # Adaptive step sizes per dimension
            steps = np.full(dim, 0.5)
            no_improve = 0

            while calls[0] < call_limit and time.time() < deadline - 10.0:
                improved = False
                for d in range(dim):
                    if calls[0] >= call_limit or time.time() >= deadline - 10.0:
                        break
                    for sign in [1.0, -1.0]:
                        if calls[0] >= call_limit or time.time() >= deadline - 10.0:
                            break
                        x_new = x.copy()
                        x_new[d] += sign * steps[d]
                        x_new[d] = np.clip(x_new[d], -5, 5)
                        f_new = F(x_new)
                        if f_new < fx:
                            x = x_new
                            fx = f_new
                            if fx < best_f:
                                best_f, best_x = fx, x.copy()
                            improved = True
                            steps[d] *= 0.9
                        else:
                            steps[d] *= 0.7
                            if steps[d] < 1e-10:
                                steps[d] = 1e-10

                if not improved:
                    no_improve += 1
                    steps *= 0.5
                    if np.all(steps < 1e-12) or no_improve > 20:
                        break
                else:
                    no_improve = 0

            return x, fx

        # ---- CMA-ES single run ----
        def cma_run(x0, f0, call_limit):
            nonlocal best_x, best_f
            n = dim
            m = int(4 + 3 * np.log(n))
            sigma = 0.5
            C = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)

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

            B = None
            D = None

            def update_ds():
                nonlocal B, D
                ev, V = np.linalg.eigh(C)
                ev = np.maximum(ev, 1e-20)
                D = np.sqrt(ev)
                B = V

            update_ds()

            x = x0.copy()
            fx = f0
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            while calls[0] < call_limit and time.time() < deadline - 15.0:
                # Sample offspring
                z = rng.standard_normal((m, n))
                off = z @ (B * D).T
                off *= sigma
                y = x[None, :] + off
                y = np.clip(y, -5, 5)

                # Evaluate
                fy = np.empty(m)
                for i in range(m):
                    if calls[0] >= call_limit or time.time() >= deadline - 15.0:
                        break
                    fy[i] = F(y[i])
                if calls[0] >= call_limit:
                    break

                # Sort
                order = np.argsort(fy)

                # Update best
                if fy[order[0]] < best_f:
                    best_f = fy[order[0]]
                    best_x = y[order[0]].copy()

                # New mean
                mean_new = np.sum(w[:, None] * y[order[:mu]], axis=0)

                # Evolution path
                diff = mean_new - x
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (
                    B @ (D ** -1) @ (diff / sigma)
                )

                # Covariance update
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls[0] - m))) < chi_n * (1.4 + 1 / n):
                    pc = (1 - c_c) * pc + c_c * (diff / sigma)
                    rank_c = np.outer(pc, pc)
                else:
                    rank_c = None

                rank_mu = np.sum(
                    w[:, None, None] * np.outer(y[order[:mu]] - x, y[order[:mu]] - x),
                    axis=0
                ) / sigma ** 2

                C = (1 + c_1 + c_mu - c_1 - c_mu) * C + c_1 * (rank_c if rank_c is not None else np.outer(diff / sigma, diff / sigma)) + c_mu * rank_mu

                # Sigma update
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(1e-12, min(10.0, sigma))

                x = mean_new
                update_ds()

            return x, fx

        # ---- Phase 1: CMA-ES for global exploration ----
        cma_limit = int(budget * 0.55)
        cma_run(best_x, best_f, cma_limit)

        # ---- Phase 2: Coordinate descent polish on CMA-ES result ----
        cd_limit = int(budget * 0.75)
        cd_result = coord_descent_polish(best_x, best_f, cd_limit)

        # ---- Phase 3: Multi-start restarts to escape local minima ----
        n_restarts = 3
        restart_budget = max(0, budget - calls[0] - 20)
        per_restart = restart_budget // max(1, n_restarts)

        for r in range(n_restarts):
            if calls[0] >= budget or time.time() >= deadline - 10.0:
                break

            # Start near best with moderate perturbation
            x_restart = best_x + rng.normal(0, 1.0, dim)
            x_restart = np.clip(x_restart, -5, 5)
            f_restart = F(x_restart)

            # Quick CMA-ES from this start
            local_limit = min(calls[0] + per_restart, int(budget * 0.9))
            if f_restart < best_f:
                best_f, best_x = f_restart, x_restart.copy()
            cma_run(x_restart, f_restart, local_limit)

            # Polish with coordinate descent
            cd_limit2 = min(calls[0] + per_restart // 2, budget - 10)
            coord_descent_polish(best_x, best_f, cd_limit2)

        # ---- Final fine polish ----
        if calls[0] < budget and time.time() < deadline - 5.0:
            coord_descent_polish(best_x, best_f, budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
