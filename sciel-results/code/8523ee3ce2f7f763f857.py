# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Global scan + CMA-ES (covariance-adapting) + coordinate polish."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def _sobol(self, n, dim, rng):
        pts = np.zeros((n, dim))
        for j in range(dim):
            base = j + 2
            for i in range(n):
                inv = 1.0 / base
                f = base
                while f <= i:
                    f *= base
                inv *= f
                pts[i, j] = inv
        pts = np.roll(pts, np.arange(1, n), axis=0)
        pts += rng.uniform(0, 1.0 / n, size=(n, dim))
        return pts

    def _cmaes(self, func, x0, fx0, budget, deadline, rng):
        n = self.dim
        calls = 1
        x = x0.copy()
        best_x, best_f = x.copy(), fx0
        sigma = 0.8
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        cs = (n + 2) / (n + 8)
        ds = cs + 2.0 / (n + 4)
        cmu = 2 / ((n + 4.0) ** 2 + n + 2)
        mu = n // 2 + 1
        w = np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / w.sum()
        chiN = np.sqrt(n) * (1 - 1 / (4.0 * n) + 1 / (21.0 * n * n))
        gen = 0

        while calls < budget:
            if time.time() >= deadline:
                break
            gen += 1
            B = np.linalg.cholesky(C)
            Y = rng.standard_normal((mu, n))
            P = Y @ B.T
            P = P / np.maximum(P.std(axis=1, keepdims=True), 1e-12)
            pop = x[None, :] + sigma * P
            vals = np.empty(mu)
            for i in range(mu):
                if time.time() >= deadline:
                    break
                vals[i] = func(np.clip(pop[i], -5.0, 5.0))
                calls += 1
            idx = np.argsort(vals)
            Ys = Y[idx]
            old_mean = x.copy()
            x = x + sigma * (w @ Ys)
            invB = np.linalg.inv(B)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invB @ (x - old_mean))
            norm_ps = np.linalg.norm(ps)
            if norm_ps / np.sqrt(1 - (1 - cs) ** (2 * gen)) < (1.4 + cs) * chiN:
                pc = (1 - ds) * pc + ds * (ps / sigma)
                hsig = 1.0
            else:
                hsig = 0.0
            C = (1 + cmu * (hsig - 1) - cmu) * C
            C += cmu * (np.outer(ps, ps) / sigma ** 2 + (pc - np.outer(pc, pc)))
            C += cmu * np.sum(w[:mu, None] * np.outer(Ys, Ys), axis=0)
            C = 0.5 * (C + C.T)
            try:
                s, logdet = np.linalg.slogdet(C)
                if s > 0:
                    C = C / np.exp(logdet / n)
            except np.linalg.LinAlgError:
                C = np.eye(n)
            # 1/5 success rule for sigma
            if vals[0] < best_f:
                best_f = vals[0]
                best_x = np.clip(pop[idx[0]], -5.0, 5.0)
                sigma *= 1.2
            else:
                sigma *= 0.9
            sigma = min(max(sigma, 1e-4), 10.0)
        return best_x, best_f, calls

    def _polish(self, func, x, fx, budget, deadline, rng):
        n = self.dim
        calls = 0
        for _ in range(40):
            if calls >= budget or time.time() >= deadline:
                break
            improved = False
            for d in range(n):
                if calls >= budget or time.time() >= deadline:
                    break
                step = max(1e-3, 0.05 * abs(x[d]) + 0.05)
                for s in (step, -step, step * 0.3, -step * 0.3):
                    if calls >= budget or time.time() >= deadline:
                        break
                    x2 = x.copy()
                    x2[d] = np.clip(x[d] + s, -5.0, 5.0)
                    f2 = func(x2)
                    calls += 1
                    if f2 < fx:
                        x, fx = x2, f2
                        improved = True
                        break
            if not improved:
                break
        return x, fx, calls

    def __call__(self, func):
        n = self.dim
        t0 = time.time()
        deadline = t0 + 300.0
        rng = np.random.default_rng(0)
        budget = self.budget
        calls = 0

        n_scan = min(60, budget // 8)
        u = self._sobol(n_scan, n, rng)
        pts = -5.0 + 10.0 * u
        best_x, best_f = None, np.inf
        for i in range(n_scan):
            if time.time() >= deadline:
                break
            x = np.clip(pts[i], -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f, best_x = f, x.copy()
        for _ in range(4):
            if time.time() >= deadline:
                break
            x = np.clip(rng.uniform(-5, 5, n), -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f, best_x = f, x.copy()

        remaining = budget - calls
        if remaining > 10 and time.time() < deadline:
            x2, f2, c2 = self._cmaes(func, best_x, best_f, remaining, deadline, rng)
            calls += c2
            if f2 < best_f:
                best_f, best_x = f2, x2

        remaining = budget - calls
        if remaining > 5 and time.time() < deadline:
            x3, f3, c3 = self._polish(func, best_x, best_f, remaining, deadline, rng)
            calls += c3
            if f3 < best_f:
                best_f, best_x = f3, x3

        return best_x, best_f
# EVOLVE-BLOCK-END
