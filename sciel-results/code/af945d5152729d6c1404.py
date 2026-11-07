"""Optimizer for BBOB-style f19 (Griewank-Rosenbrock) on [-5,5]^5."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270.0  # safety margin under 300s wall-clock

        # ---- time-aware call wrapper ----
        call_times = []
        n_calls = 0

        def call(x):
            nonlocal n_calls
            t0 = time.time()
            v = func(np.asarray(x, dtype=float))
            dt = time.time() - t0
            call_times.append(dt)
            n_calls += 1
            return float(v)

        def remaining_calls():
            # estimate average cost from recent calls
            if call_times:
                recent = call_times[-20:] if len(call_times) >= 20 else call_times
                avg = float(np.mean(recent))
            else:
                avg = 1e-4
            avg = max(avg, 1e-6)
            left = deadline - time.time() - 2.0  # 2s safety
            return int(max(0, left / avg))

        # ---- Nelder-Mead local polish (cheap, no derivatives) ----
        def nm_polish(x0, f0, max_iter):
            n = dim
            eps = 0.1
            simplex = [np.array(x0, dtype=float)]
            simplex.append(np.array(x0, dtype=float) + eps * np.eye(n)[0])
            # build full simplex
            simplex = [np.array(x0, dtype=float)]
            for i in range(n):
                pt = np.array(x0, dtype=float)
                pt[i] += eps
                simplex.append(pt)
            vals = [call(p) for p in simplex]
            best = 0
            for _ in range(max_iter):
                if remaining_calls() < n + 3:
                    break
                order = np.argsort(vals)
                simplex = [simplex[i] for i in order]
                vals = [vals[i] for i in order]
                best = 0
                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + 1.0 * (centroid - simplex[-1])  # reflect
                fr = call(xr)
                if fr < vals[0]:
                    # expand
                    xe = centroid + 2.0 * (xr - centroid)
                    fe = call(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        vals[-1] = fr
                elif fr < vals[-2]:
                    simplex[-1] = xr
                    vals[-1] = fr
                else:
                    # contract
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    fc = call(xc)
                    if fc < vals[-1]:
                        simplex[-1] = xc
                        vals[-1] = fc
                    else:
                        # shrink toward best
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            vals[i] = call(simplex[i])
            order = np.argsort(vals)
            return simplex[order[0]], vals[order[0]]

        # ---- CMA-ES (small population) ----
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
            best_x, best_f = None, np.inf
            for it in range(max_iter):
                if remaining_calls() < pop * 2 + n + 5:
                    break
                # sample
                Y = np.array([rng.standard_normal(n) for _ in range(pop)])
                X = mean[None, :] + sigma * (Y @ (B * D[None, :]))
                F = np.array([call(x) for x in X])
                order = np.argsort(F)
                X = X[order]
                F = F[order]
                # track best
                if F[0] < best_f:
                    best_f = F[0]
                    best_x = X[0].copy()
                # update
                y_w = np.sum(w[:, None] * X[:mu], axis=0)
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (B @ (D @ (y_w / sigma)))
                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (it + 1)))) < (1.4 + 2.0 / (n + 1)) * chiN
                pc = (1.0 - cc) * pc + hsig * cc * y_w
                y_cov = (y_w / sigma)
                rank_mu = np.outer(y_cov, y_cov)
                rank_one = np.outer(pc, pc)
                B, D = cmaes._update_BD(B, D, pc, y_cov, rank_mu, rank_one, c1, cmu, cc, cs, mueff, n, chiN, sigma)
                mean = mean + sigma * y_w
                # step-size adaptation
                sigma = sigma * np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1.0))
                sigma = min(sigma, 1e4)
            return best_x, best_f

        @staticmethod
        def _update_BD(B, D, pc, y_cov, rank_mu, rank_one, c1, cmu, cc, cs, mueff, n, chiN, sigma):
            # simplified eigen-based update
            H = (1.0 - c1 - cmu) * np.eye(n) + c1 * rank_one + cmu * rank_mu
            # apply to B, D: (B D) H (B D)^T -> new B, D
            M = (B * D[None, :]) @ H @ (B * D[None, :]).T
            try:
                eigvals, eigvecs = np.linalg.eigh(M)
                eigvals = np.maximum(eigvals, 1e-20)
                B = eigvecs
                D = np.sqrt(eigvals)
            except np.linalg.LinAlgError:
                pass
            return B, D

        rng = np.random.default_rng(12345)

        # ---- initial point: center + cheap multi-probe ----
        x0 = np.zeros(dim)
        f0 = call(x0)
        best_x, best_f = x0.copy(), f0

        # probe a few deterministic + random starts to get a better initial
        starts = [np.zeros(dim)]
        starts.append(np.full(dim, 1.0))
        starts.append(np.full(dim, -1.0))
        for _ in range(8):
            starts.append(rng.uniform(-4, 4, dim))
        for s in starts:
            if remaining_calls() < 5:
                break
            fs = call(s)
            if fs < best_f:
                best_f = fs
                best_x = s.copy()

        # ---- phase 1: local polish on best start (fast early progress) ----
        if remaining_calls() > 30:
            px, pf = nm_polish(best_x, best_f, max_iter=25)
            if pf < best_f:
                best_f, best_x = pf, px

        # ---- phase 2: CMA-ES global search from best point ----
        if remaining_calls() > 50:
            pop = min(15, max(6, 2 * dim + 2))
            sigma0 = 1.5
            cx, cf = cmaes(best_x, sigma0, pop, max_iter=remaining_calls())
            if cf < best_f:
                best_f, best_x = cf, cx

        # ---- phase 3: final local polish on global best ----
        if remaining_calls() > 20:
            px, pf = nm_polish(best_x, best_f, max_iter=40)
            if pf < best_f:
                best_f, best_x = pf, px

        # ---- phase 4: if time remains, a few more multi-start CMA-ES ----
        while remaining_calls() > 60:
            s = best_x + 0.5 * rng.standard_normal(dim)
            s = np.clip(s, -5, 5)
            fs = call(s)
            if fs < best_f:
                best_f, best_x = fs, s
            pop = min(12, max(6, 2 * dim + 2))
            cx, cf = cmaes(s, 1.0, pop, max_iter=remaining_calls())
            if cf < best_f:
                best_f, best_x = cf, cx
            if remaining_calls() > 15:
                px, pf = nm_polish(best_x, best_f, max_iter=20)
                if pf < best_f:
                    best_f, best_x = pf, px

        return best_x, best_f
