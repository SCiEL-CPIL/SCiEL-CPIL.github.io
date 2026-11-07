import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin vs 330s wall-clock kill

        call_count = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf

        def call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget or time.time() > deadline:
                return np.inf
            call_count += 1
            xv = np.asarray(x, dtype=float)
            xv = np.clip(xv, -5.0, 5.0)
            try:
                f = float(func(xv))
            except Exception:
                return np.inf
            if not np.isfinite(f):
                return np.inf
            if f < best_f:
                best_f = f
                best_x = xv.copy()
            return f

        dim = self.dim
        budget = self.budget
        rng = np.random.RandomState(12345)

        # ---------------- Phase 1: Global exploration via (1+1)-ES restarts ----------------
        explore_budget = int(0.62 * budget)
        n_starts = 12
        per_start = max(15, (explore_budget - call_count) // max(1, n_starts))

        for _ in range(n_starts):
            if call_count >= explore_budget or time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, size=dim)
            f = call(x)
            sigma = 1.5
            for _ in range(per_start):
                if call_count >= explore_budget or time.time() > deadline:
                    break
                step = sigma * rng.randn(dim)
                y = x + step
                fy = call(y)
                if fy < f:
                    x = y
                    f = fy
                    sigma *= 1.25
                else:
                    sigma *= 0.8
                sigma = min(sigma, 3.0)
                if sigma < 1e-4:
                    break

        # ---------------- Phase 2: Local refinement via CMA-ES ----------------
        def cma_es(x0, max_iter, sigma0):
            nonlocal call_count, best_x, best_f
            d = dim
            x0 = np.asarray(x0, dtype=float).copy()
            x0 = np.clip(x0, -5.0, 5.0)
            m = max(4, (d + 1) // 2)
            lam = 2 * m
            w = np.log(m + 0.5) - np.log(np.arange(1, lam + 1))
            w = w / w.sum()
            mueff = int(np.sum(w[:m]) ** 2 / np.sum(w ** 2))
            mu = w[:mueff]
            mu = mu / mu.sum()
            c1 = (2 + 2.0 / (d + 1.3)) / ((d + 1) ** 2 + 2)
            cmu = (2 + np.sum(mu ** 2) / np.sum(mu ** 2) - 2) / ((d + 2) ** 2 + 2)
            cmu = min(cmu, 2 * (1 - c1) / (d + 1))
            cs = (mueff + 2) / (d + mueff + 5)
            ds = 1 + cs + 2 * max(0.0, (mueff - 1) / (d + 1) - 1)
            cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
            chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
            sigma = sigma0
            C = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            best = call(x0)
            for _ in range(max_iter):
                if call_count >= budget or time.time() > deadline:
                    break
                if best < best_f:
                    best_f = best
                    best_x = x0.copy()
                B, D = np.linalg.eigh(C)
                B = B @ np.diag(1.0 / np.sqrt(np.maximum(D, 1e-20)))
                z = rng.randn(lam, d)
                pop = x0[None, :] + sigma * (z @ B.T)
                pop = np.clip(pop, -5.0, 5.0)
                fvals = np.array([call(p) for p in pop])
                order = np.argsort(fvals, kind='stable')
                y = pop[order]
                yw = y[:mueff]
                mean_y = np.sum(mu[:, None] * yw, axis=0)
                zc = np.sum(mu[:, None] * yw, axis=0)
                pc = (1 - cc) * pc + cc * (sigma * zc)
                hsig = (np.sqrt(1 - (1 - cc) ** (2 * (max_iter - call_count) + 1)) <
                        1 + 2.0 / (d + 1))
                ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mueff) * (pc / sigma)
                delta_hsig = 1.0 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2)
                                     < (1.4 + 2.0 / (d + 1)) * chiN) else 0.0
                C = ((1 + c1 * delta_hsig * (mueff + 2) - c1 - cmu) * C
                     + c1 * np.outer(pc, pc)
                     + cmu * np.sum(mu[:, None, None] * np.outer(yw, yw), axis=0))
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
                sigma = min(max(sigma, 1e-12), 50.0)
                x0 = x0 + c1 * pc + cmu * mean_y
                x0 = np.clip(x0, -5.0, 5.0)
                if np.linalg.norm(x0 - y[0]) < (1e-12 + 1e-9 * sigma):
                    break
            if best < best_f:
                best_f = best
                best_x = x0.copy()

        # Try a few CMA-ES refinements, restarting on stagnation if budget allows
        n_cma = 2
        for k in range(n_cma):
            if call_count >= budget or time.time() > deadline:
                break
            max_iter = max(50, (budget - call_count) // (dim + 2))
            cma_es(best_x, max_iter, sigma0=0.5)

        return best_x, best_f
