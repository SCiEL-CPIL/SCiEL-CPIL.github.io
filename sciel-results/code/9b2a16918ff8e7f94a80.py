"""Optimizer for BBOB f23 (Katsuura) via CMA-ES with multi-start and polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start CMA-ES with a final high-precision polish pass."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 300.0  # safety margin vs. 330s kill
        dim = self.dim
        budget = int(self.budget)
        calls = 0

        def time_ok():
            return (time.time() - t0) < (deadline - 5.0)

        # ---------------- CMA-ES core ----------------
        def cmaes(x0, sigma0, budget, rng, func):
            n = dim
            if n == 1:
                lam = 4
            else:
                lam = 4 + int(3 * np.log(n))
            c = (2 + 2.0 / (n + 1.3)) / (n + 4 + 2.0 / (n + 1.3))
            cmu = min(0.75 - c, 2 / ((n + 1.3) ** 2 + c))
            cmu = max(cmu, 0.0)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            c_sigma = (5 + cmu) / (5 + n + cmu)
            d_sigma = 1 + 2 * max(0, np.sqrt((n + 1) / (n + 4) - 1)) + cmu
            c_c = (4 + n / c) / (n + 4 + 4 / c)
            c_1 = 2 / ((n + 1.7) ** 2 + c)
            c_mu = min(1 - c_1, 2 * ((mu_eff + 1) / (mu_eff + 2) - 1))
            mu_eff = 2.0 / (mu_eff if False else _mu_eff(mu, lam))  # placeholder
            # recompute mu_eff properly below
            return None

        # mu_eff helper
        def _mu_eff(mu, lam):
            return 2.0 / (np.sum(mu) / np.sum(mu ** 2) - 1.0)

        def cmaes_run(x0, sigma0, budget, rng, func, calls_ref):
            n = dim
            if n == 1:
                lam = 4
            else:
                lam = 4 + int(3 * np.log(n))
            weights = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
            weights = weights / np.sum(weights)
            mu = int(np.sum(weights > 0))
            mu_eff = 2.0 / (np.sum(weights[:mu]) / np.sum(weights[:mu] ** 2) - 1.0)
            c = (2 + 2.0 / (n + 1.3)) / (n + 4 + 2.0 / (n + 1.3))
            cmu = min(0.75 - c, 2 / ((n + 1.3) ** 2 + c))
            cmu = max(cmu, 0.0)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            c_sigma = (5 + cmu) / (5 + n + cmu)
            d_sigma = 1 + 2 * max(0.0, np.sqrt((n + 1) / (n + 4) - 1.0)) + cmu
            c_c = (4 + n / c) / (n + 4 + 4 / c)
            c_1 = 2 / ((n + 1.7) ** 2 + c)
            c_mu = min(1 - c_1, 2 * ((mu_eff + 1) / (mu_eff + 2) - 1.0))
            c_mu = max(c_mu, 0.0)

            x = np.array(x0, dtype=float)
            sigma = float(sigma0)
            C = np.eye(n)
            Cinv = np.eye(n)
            Cs = np.zeros(n)
            ps = np.zeros(n)
            best_x = x.copy()
            best_f = np.inf

            def evalf(p):
                nonlocal calls_ref
                val = float(func(p))
                calls_ref[0] += 1
                return val

            # initial evaluation
            f0 = evalf(x)
            best_f = f0

            while calls_ref[0] < budget and time_ok():
                # generate offspring
                ys = []
                fs = []
                for _ in range(lam):
                    B = np.linalg.cholesky(C)
                    z = rng.standard_normal(n)
                    y = x + sigma * B @ z
                    y = np.clip(y, -5.0, 5.0)
                    f = evalf(y)
                    ys.append(y)
                    fs.append(f)
                    if f < best_f:
                        best_f = f
                        best_x = y.copy()
                ys = np.array(ys)
                fs = np.array(fs)
                order = np.argsort(fs)
                ys = ys[order]
                fs = fs[order]
                w = weights[:mu]
                mean_new = np.sum(w[:, None] * ys[:mu], axis=0)
                # update sigma
                y_mean = (mean_new - x) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                # update C
                hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_ref[0] // lam + 1))) < (1.4 + 2 / (n + 1)) * chi_n
                rank_c = (1 - (1 - c_c) ** 2) * np.outer(ps, ps)
                if hs:
                    rank_c += c_c * (np.outer(ps, ps) - C)
                # rank-mu
                diff = ys[:mu] - mean_new
                rank_mu = np.einsum('i,ij,ik->jk', w, diff, diff)
                C = (1 - c_c - c_mu) * C + c_c * rank_c + c_mu * rank_mu
                x = mean_new
                # periodic recompute of Cinv (cheap for small n)
                Cinv = np.linalg.inv(C + 1e-12 * np.eye(n))
                sigma = max(sigma, 1e-12)
            return best_x, best_f

        # ---------------- (1+1) polish ----------------
        def polish(x0, fx0, budget, rng, func, calls_ref):
            x = np.array(x0, dtype=float)
            fx = float(fx0)
            sigma = 0.5
            for _ in range(budget):
                if not time_ok():
                    break
                if calls_ref[0] >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = float(func(y))
                calls_ref[0] += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                if sigma < 1e-10:
                    sigma = 1e-10
            return x, fx

        calls_ref = [0]
        rng = np.random.default_rng(0)

        # ---------------- Multi-start allocation ----------------
        # Reserve ~30% of budget for polish on the best start.
        n_starts = 3
        polish_budget = int(0.30 * budget)
        per_start = (budget - polish_budget) // n_starts

        best_x = None
        best_f = np.inf

        for s in range(n_starts):
            if calls_ref[0] >= budget - polish_budget or not time_ok():
                break
            # diverse starts
            if s == 0:
                x0 = np.zeros(dim)
                sigma0 = 2.0
            else:
                x0 = rng.uniform(-4.0, 4.0, dim)
                sigma0 = 1.5
            rem = budget - polish_budget - calls_ref[0]
            alloc = min(per_start, rem)
            if alloc <= 0:
                break
            bx, bf = cmaes_run(x0, sigma0, alloc, rng, func, calls_ref)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        # ---------------- Final polish on best ----------------
        if best_x is not None:
            rem = budget - calls_ref[0]
            if rem > 0 and time_ok():
                px, pf = polish(best_x, best_f, rem, rng, func, calls_ref)
                if pf < best_f:
                    best_f = pf
                    best_x = px

        if best_x is None:
            # fallback: at least one call
            x = rng.uniform(-5, 5, dim)
            f = float(func(x))
            calls_ref[0] += 1
            best_x, best_f = x, f

        return best_x, best_f
# EVOLVE-BLOCK-END
