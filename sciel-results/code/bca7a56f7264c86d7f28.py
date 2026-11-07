"""Rewritten optimizer for BBOB f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive restart for Lunacek bi-Rastrigin (f24)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        best_x_global = None
        fbest_global = np.inf
        calls_total = 0

        def time_ok():
            return time.time() < deadline - 5.0

        def calls_ok():
            return calls_total < budget

        # Multi-start CMA-ES: try several starting points with different sigmas
        # For bi-Rastrigin, the global minimum is typically away from origin
        # Start with a set of diverse initial points

        start_points = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.array([3.0, -3.0, 2.0, -2.0, 1.0])[:dim],
            np.array([-2.0, 3.0, -1.0, 2.0, -3.0])[:dim],
            np.array([1.5, 1.5, -1.5, -1.5, 1.5])[:dim],
        ]
        # Add a couple random starts
        for _ in range(2):
            sp = np.random.uniform(-3, 3, size=dim)
            start_points.append(sp)

        sigma_inits = [2.5, 2.0, 2.5, 1.5, 1.5, 1.0, 1.5, 1.0]

        # Allocate budget across runs: more budget to earlier/better runs
        n_runs = min(len(start_points), 6)
        # Give first run ~40%, second ~25%, rest share remaining
        remaining_budget = budget
        run_budgets = []
        for i in range(n_runs):
            if i == 0:
                rb = int(budget * 0.35)
            elif i == 1:
                rb = int(budget * 0.25)
            else:
                rb = int(remaining_budget / max(1, n_runs - i))
            rb = max(30, min(rb, remaining_budget - 20))
            run_budgets.append(rb)
            remaining_budget -= rb

        # Ensure we don't exceed budget
        total_planned = sum(run_budgets)
        if total_planned > budget - 10:
            scale = (budget - 10) / total_planned
            run_budgets = [max(30, int(b * scale)) for b in run_budgets]

        for i in range(n_runs):
            if not calls_ok() or not time_ok():
                break

            sp = start_points[i]
            sig = sigma_inits[i % len(sigma_inits)]
            rb = run_budgets[i]

            # Skip if this start is too close to an already-tried start
            if best_x_global is not None and np.linalg.norm(sp - best_x_global) < 0.5:
                continue

            x0, sigma, best_x, fbest, calls_used = self._run_cmaes(
                func, rb, dim,
                start_point=sp,
                sigma_init=sig,
                deadline=deadline
            )
            calls_total += calls_used
            if fbest < fbest_global:
                fbest_global = fbest
                best_x_global = best_x

        # Final local polish around the best point
        if calls_ok() and time_ok() and best_x_global is not None:
            remaining = budget - calls_total
            if remaining >= 15:
                best_x_global, fbest_global = self._polish(
                    func, best_x_global, fbest_global, dim, remaining, deadline
                )
                calls_total = budget  # polish uses up to remaining

        return best_x_global, fbest_global

    def _run_cmaes(self, func, budget, dim, start_point=None, sigma_init=1.0, deadline=None):
        # CMA-ES parameters
        lambda_ = 15
        mu = lambda_ // 2

        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initialization
        if start_point is None:
            x0 = np.zeros(dim)
        else:
            x0 = start_point.copy()

        sigma = sigma_init

        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        def sqrtm(C_mat):
            try:
                return np.linalg.cholesky(C_mat)
            except np.linalg.LinAlgError:
                vals, vecs = np.linalg.eigh(C_mat)
                vals = np.maximum(vals, 1e-10)
                return vecs @ np.diag(np.sqrt(vals)) @ vecs.T

        B = sqrtm(C)

        fbest = func(x0)
        best_x = x0.copy()
        calls = 1

        # Track no-improvement streak for early stopping
        no_improve_calls = 0
        min_no_improve = 40  # stop if no improvement for this many calls

        while calls < budget:
            if deadline is not None and time.time() > deadline - 3:
                break

            # Generate population
            ys = []
            fs = []

            for i in range(lambda_):
                if calls >= budget:
                    break
                if deadline is not None and time.time() > deadline - 3:
                    break

                z = np.random.standard_normal(dim)
                y = x0 + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                ys.append(y)

                fy = func(y)
                fs.append(fy)
                calls += 1

                if fy < fbest:
                    fbest = fy
                    best_x = y.copy()
                    no_improve_calls = 0
                else:
                    no_improve_calls += 1

            if calls >= budget or len(ys) < lambda_:
                break

            # Early stopping if no improvement
            if no_improve_calls >= min_no_improve:
                break

            fs_arr = np.array(fs)
            order = np.argsort(fs_arr)
            ys = [ys[i] for i in order]
            fs_arr = fs_arr[order]

            # Update mean
            x_new = sum(w[i] * ys[i] for i in range(mu))

            # Recompute C_inv_sqrt
            vals, vecs = np.linalg.eigh(C)
            vals_inv_sqrt = 1.0 / np.sqrt(np.maximum(vals, 1e-10))
            C_inv_sqrt = (vecs * vals_inv_sqrt) @ vecs.T

            # Update sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                 C_inv_sqrt @ (x_new - x0) / sigma

            E_norm_ps = np.sqrt(dim * (1 - (1 - c_sigma) ** 2))
            h_sigma = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1))) < \
                        1.4 * E_norm_ps else 0.0

            sigma *= np.exp(c_sigma / dim * (np.linalg.norm(ps) / E_norm_ps - 1) * h_sigma)

            # Update covariance
            pc = (1 - c_c) * pc + c_c * (x_new - x0) / sigma
            yk = x_new - x0
            rank1 = np.outer(yk, yk)
            rankmu = sum(w[i] * np.outer(ys[i] - x0, ys[i] - x0) for i in range(mu))

            C = (1 + c_1 * (c_mu * mu_eff / dim - 1) - c_1 - c_mu) * C \
                + c_1 * (pc @ pc + (1 - (1 - c_c) ** 2) * C) \
                + c_mu * rankmu

            C = (C + C.T) / 2

            x0 = x_new
            B = sqrtm(C)

            # Convergence checks
            if sigma < 1e-8:
                break
            if calls > 30 and sigma < 1e-6:
                break

        return x0, sigma, best_x, fbest, calls

    def _polish(self, func, x_init, f_init, dim, remaining_budget, deadline=None):
        """Local coordinate search with decreasing step sizes."""
        if remaining_budget < 15:
            return x_init, f_init

        best_x = x_init.copy()
        best_f = f_init
        calls_made = 0
        max_calls = min(remaining_budget, 150)

        step = 0.3
        for _ in range(max_calls // dim + 1):
            improved = False
            for d in range(dim):
                for sign in [1, -1]:
                    if calls_made >= max_calls:
                        break
                    if deadline is not None and time.time() > deadline - 2:
                        break
                    x_trial = best_x.copy()
                    x_trial[d] += sign * step
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls_made += 1
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = x_trial.copy()
                        improved = True
                if calls_made >= max_calls:
                    break
            if not improved:
                step *= 0.5
                if step < 1e-5:
                    break
            if calls_made >= max_calls:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
