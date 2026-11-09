"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES + Rosenbrock gradient boost + coordinate descent for f9."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _rosenbrock_grad(self, x):
        """Analytic gradient of standard Rosenbrock."""
        n = len(x)
        g = np.zeros(n)
        g[0] = -2.0 * (1.0 - x[0])
        for i in range(1, n):
            g[i] = 4.0 * x[i] * (x[i - 1] - 1.0) - 2.0 * (1.0 - x[i - 1])
        return g

    def _coord_descent(self, x, func, budget_remaining, deadline, rng):
        """Simple coordinate descent with line search along each axis."""
        n = len(x)
        fx = func(x)
        calls_used = 0
        step = 0.5
        for _ in range(3):
            if calls_used >= budget_remaining or time.time() > deadline:
                break
            for i in range(n):
                if calls_used >= budget_remaining or time.time() > deadline:
                    break
                for s in [step, step * 0.5, step * 0.25, -step, -step * 0.5, -step * 0.25]:
                    if calls_used >= budget_remaining or time.time() > deadline:
                        break
                    y = x.copy()
                    y[i] = np.clip(y[i] + s, -5, 5)
                    fy = func(y)
                    calls_used += 1
                    if fy < fx:
                        x, fx = y, fy
                        break
        return x, fx, calls_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280

        rng = np.random.default_rng(2024)
        n = self.dim
        budget = self.budget
        calls_used = 0

        def call_func(x):
            nonlocal calls_used
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            return fx

        # --- Phase 1: Diverse initialization ---
        init_points = [
            np.zeros(n),
            np.ones(n) * 1.0,
            np.ones(n) * -1.0,
            np.ones(n) * 2.0,
            np.ones(n) * -2.0,
            np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([-1.0, -1.0, -1.0, -1.0, -1.0]),
        ]
        for _ in range(8):
            init_points.append(rng.uniform(-4, 4, n))

        best_x = None
        best_f = np.inf
        for init in init_points:
            if calls_used >= budget or time.time() > deadline:
                break
            x = np.clip(init, -5, 5)
            fx = call_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = call_func(best_x)

        # --- Phase 2: CMA-ES with gradient boost ---
        x_mean = best_x.copy()
        sigma = 1.0
        C = np.eye(n)
        mu = max(2, int(np.floor(n / 2)))
        lam = 2 * mu
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        ps = np.zeros(n)
        pc = np.zeros(n)
        iter_count = 0
        max_cma_iters = 150

        while calls_used < budget and time.time() < deadline and iter_count < max_cma_iters:
            iter_count += 1
            L = np.linalg.cholesky(C)

            # Sample
            ys, xs = [], []
            for _ in range(lam):
                z = rng.standard_normal(n)
                y = L @ z
                x_i = np.clip(x_mean + sigma * y, -5, 5)
                ys.append(y)
                xs.append(x_i)

            # Evaluate
            fs = []
            for x_i in xs:
                if calls_used >= budget or time.time() > deadline:
                    break
                fs.append(call_func(x_i))

            if len(fs) < mu:
                break

            # Sort
            idx = np.argsort(fs)
            xs = [xs[i] for i in idx]
            ys = [ys[i] for i in idx]
            fs = [fs[i] for i in idx]

            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()

            # New mean
            x_new = np.zeros(n)
            for i in range(mu):
                x_new += weights[i] * xs[i]

            # Evolution paths
            y_mean = np.zeros(n)
            for i in range(mu):
                y_mean += weights[i] * ys[i]

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (L.T @ y_mean)
            ps_norm = np.linalg.norm(ps)
            h_sigma = ps_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (iter_count + 1))) / chi_n
            h_sig = min(1.0, h_sigma / (1 + 1e-9))

            pc = (1 - c_c) * pc + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (L.T @ y_mean)

            # Covariance update
            delta_h = (1 - h_sig) * c_c * (2 - c_c)
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                zi = L.T @ ys[i]
                rank_mu += weights[i] * np.outer(zi, zi)

            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * (rank_one - C) + c_mu * rank_mu
            C = 0.5 * (C + C.T)

            # Sigma update
            sigma *= np.exp(c_sigma / chi_n * (ps_norm / chi_n - 1))
            x_mean = x_new

            # Gradient boost
            if calls_used < budget and time.time() < deadline:
                g = self._rosenbrock_grad(x_mean)
                g_norm = np.linalg.norm(g)
                if g_norm > 1e-8:
                    for alpha in [sigma * 0.1, sigma * 0.05, sigma * 0.01]:
                        if calls_used >= budget or time.time() > deadline:
                            break
                        x_test = np.clip(x_mean - alpha * g / g_norm, -5, 5)
                        f_test = call_func(x_test)
                        if f_test < best_f:
                            best_f = f_test
                            best_x = x_test.copy()
                            x_mean = x_test.copy()
                            break

            # Coordinate descent polish
            if iter_count % 5 == 0 and calls_used < budget and time.time() < deadline:
                x_cd, f_cd, cd_calls = self._coord_descent(
                    x_mean.copy(), func, budget - calls_used, deadline, rng
                )
                calls_used += cd_calls
                if f_cd < best_f:
                    best_f = f_cd
                    best_x = x_cd.copy()
                    x_mean = x_cd.copy()

            if sigma < 1e-6:
                break

        # --- Phase 3: Final polish ---
        if calls_used < budget and time.time() < deadline:
            x_final, f_final, _ = self._coord_descent(
                best_x.copy(), func, budget - calls_used, deadline, rng
            )
            if f_final < best_f:
                best_f = f_final
                best_x = x_final.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
