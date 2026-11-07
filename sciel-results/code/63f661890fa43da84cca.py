"""Optimizer for BBOB-style f19 (Griewank-Rosenbrock) on [-5,5]^5."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # 15s margin before 300s kill

        n_calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def call(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget or time.time() > deadline - 2.0:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            n_calls += 1
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if np.isfinite(f) and f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def remaining_calls():
            left = deadline - time.time() - 2.0
            return max(0, budget - n_calls)

        # ---- Nelder-Mead local polish ----
        def nm_polish(x0, max_iter=50):
            if remaining_calls() < dim + 5:
                return x0.copy()
            try:
                res = minimize(call, x0, method='Nelder-Mead',
                               options={'maxiter': max_iter, 'xatol': 1e-8,
                                        'fatol': 1e-14, 'adaptive': True})
                if np.isfinite(res.fun):
                    return np.clip(res.x, -5, 5)
            except Exception:
                pass
            return x0.copy()

        # ---- L-BFGS-B local search (exploits smoothness) ----
        def lbfgsb(x0, max_iter=50):
            if remaining_calls() < dim + 5:
                return x0.copy()
            try:
                res = minimize(call, x0, method='L-BFGS-B',
                               bounds=[(-5, 5)] * dim,
                               options={'maxiter': max_iter, 'gtol': 1e-8,
                                        'ftol': 1e-12})
                if np.isfinite(res.fun):
                    return np.clip(res.x, -5, 5)
            except Exception:
                pass
            return x0.copy()

        # ---- CMA-ES for global exploration ----
        def cmaes(x0, sigma0, pop, max_iter):
            n = dim
            mu = max(1, pop // 2)
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1.0 / np.sum(w ** 2)
            cc = (4.0 + mueff / (n + 4.0)) / ((n + 1.3) + (4.0 + mueff / (n + 4.0)))
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = (2.0 / ((n + 1.8) ** 2 + mueff)) * (1.0 - cc + 2.0 * min(1.0, cs / cc) * max(0.0, 1.0 - 1.0 / cc))
            cmu = (2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff)) * (1.0 - c1) * min(1.0, cs / cc)
            cmu = min(cmu, 1.0 - c1)
            chiN = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            sigma = sigma0
            mean = np.array(x0, dtype=float)
            local_best_x, local_best_f = mean.copy(), np.inf

            rng_local = np.random.default_rng(42 + int(hash(tuple(x0)) % 10000))

            for it in range(max_iter):
                if remaining_calls() < pop * 2 + n + 5:
                    break
                Y = np.array([rng_local.standard_normal(n) for _ in range(pop)])
                X = mean[None, :] + sigma * (Y @ (B * D[None, :]))
                F = np.array([call(x) for x in X])
                order = np.argsort(F)
                X = X[order]
                F = F[order]
                if F[0] < local_best_f:
                    local_best_f = F[0]
                    local_best_x = X[0].copy()

                y_w = np.sum(w[:, None] * X[:mu], axis=0)
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (B @ (D @ (y_w / sigma)))
                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (it + 1)))) < (1.4 + 2.0 / (n + 1)) * chiN
                pc = (1.0 - cc) * pc + hsig * cc * y_w
                y_cov = (y_w / sigma)
                rank_mu = np.outer(y_cov, y_cov)
                rank_one = np.outer(pc, pc)
                H = (1.0 - c1 - cmu) * np.eye(n) + c1 * rank_one + cmu * rank_mu
                M = (B * D[None, :]) @ H @ (B * D[None, :]).T
                try:
                    eigvals, eigvecs = np.linalg.eigh(M)
                    eigvals = np.maximum(eigvals, 1e-20)
                    B = eigvecs
                    D = np.sqrt(eigvals)
                except np.linalg.LinAlgError:
                    pass
                mean = mean + sigma * y_w
                sigma = sigma * np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1.0))
                sigma = min(sigma, 1e4)

            return local_best_x, local_best_f

        rng = np.random.default_rng(12345)

        # ---- Phase 1: Multi-start with L-BFGS-B (fast convergence on smooth f19) ----
        starts = [np.zeros(dim)]
        starts.append(np.full(dim, 1.0))
        starts.append(np.full(dim, -1.0))
        starts.append(np.full(dim, 2.0))
        starts.append(np.full(dim, -2.0))
        for _ in range(6):
            starts.append(rng.uniform(-4, 4, dim))
        for _ in range(3):
            starts.append(rng.uniform(-5, -3, dim))
            starts.append(rng.uniform(3, 5, dim))

        calls_per_opt = max(20, (budget - 20) // len(starts))

        for s in starts:
            if remaining_calls() < 10:
                break
            if time.time() > deadline - 5:
                break
            s = np.clip(s, -5, 5)
            call(s)  # evaluate start
            if not np.isfinite(best_f):
                continue
            # L-BFGS-B polish
            x_new = lbfgsb(s, max_iter=min(calls_per_opt // 5, 40))
            if np.isfinite(call(x_new)):
                pass  # call already updated best_f/best_x

        # ---- Phase 2: Nelder-Mead polish on global best ----
        if remaining_calls() > 20:
            x_new = nm_polish(best_x, max_iter=40)
            if np.isfinite(call(x_new)):
                pass

        # ---- Phase 3: CMA-ES global search from best point ----
        if remaining_calls() > 50:
            pop = min(12, max(6, 2 * dim + 2))
            sigma0 = 1.5
            cx, cf = cmaes(best_x, sigma0, pop, max_iter=remaining_calls())
            if np.isfinite(cf) and cf < best_f:
                best_f = cf
                best_x = cx.copy()

        # ---- Phase 4: Final L-BFGS-B + NM polish on CMA-ES result ----
        if remaining_calls() > 15:
            x_new = lbfgsb(best_x, max_iter=30)
            if np.isfinite(call(x_new)):
                pass
        if remaining_calls() > 10:
            x_new = nm_polish(best_x, max_iter=30)
            if np.isfinite(call(x_new)):
                pass

        # ---- Phase 5: If time remains, a few more multi-start L-BFGS-B ----
        while remaining_calls() > 40 and time.time() < deadline - 10:
            s = best_x + 0.5 * rng.standard_normal(dim)
            s = np.clip(s, -5, 5)
            call(s)
            x_new = lbfgsb(s, max_iter=20)
            if np.isfinite(call(x_new)):
                pass
            if remaining_calls() > 10:
                x_new = nm_polish(best_x, max_iter=15)
                if np.isfinite(call(x_new)):
                    pass

        return best_x, best_f
