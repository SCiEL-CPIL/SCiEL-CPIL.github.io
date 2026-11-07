"""Optimizer for BBOB f4 (Büche-Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Global search via L-SR1 (Latin hypercube + short (1+1)-ES) followed by
    many short CMA-ES restarts from the best basins found.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s kill

        calls = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # ---- Phase 1: Global exploration with LHS sampling + (1+1)-ES polish ----
        # Büche-Rastrigin has many local minima. We want to sample broadly,
        # then refine the best few candidates.
        n_lhs = 32
        # Latin hypercube sampling in [-5,5]^d
        lhs_points = np.zeros((n_lhs, d))
        for i in range(n_lhs):
            for j in range(d):
                lhs_points[i, j] = -5.0 + (i + rng.random()) / n_lhs * 10.0

        best_x = None
        best_f = np.inf
        candidates = []  # list of (f, x)

        for i in range(n_lhs):
            if calls >= budget or time.time() > deadline:
                break
            x = clip(lhs_points[i])
            fx = float(func(x))
            calls += 1
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f, best_x = fx, x.copy()

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Short (1+1)-ES polish from the top few candidates
        n_polish = min(6, len(candidates))
        for idx in range(n_polish):
            if calls >= budget or time.time() > deadline:
                break
            cx, cf = candidates[idx][1].copy(), candidates[idx][0]
            sigma = 0.8
            for _ in range(25):
                if calls >= budget or time.time() > deadline:
                    break
                y = clip(cx + sigma * rng.standard_normal(d))
                fy = float(func(y))
                calls += 1
                if fy < cf:
                    cx, cf = y.copy(), fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                sigma = min(sigma, 3.0)
            if cf < best_f:
                best_f, best_x = cf, cx.copy()

        # ---- Phase 2: Multiple short CMA-ES restarts from diverse basins ----
        # Use the top distinct basins (by distance) as starting points.
        starts = []
        for f_val, x_val in candidates:
            if len(starts) >= 5:
                break
            # Check distance from existing starts
            if all(np.linalg.norm(x_val - s[1]) > 1.0 for s in starts):
                starts.append((f_val, x_val.copy()))
        # Also add the globally best
        if not any(np.linalg.norm(best_x - s[1]) < 1e-6 for s in starts):
            starts.append((best_f, best_x.copy()))

        n_restarts = 5
        cma_budget_per_restart = max(10, (budget - calls) // max(1, n_restarts))

        for r_idx in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break
            if r_idx < len(starts):
                mu = starts[r_idx][1].copy()
            else:
                mu = best_x.copy()

            # CMA-ES parameters
            lam = 6
            m = lam // 2
            w = np.log(np.arange(1.0, m + 1)) - np.mean(np.log(np.arange(1.0, m + 1)))
            w = w / w.sum()
            mueff = int(1 / np.sum(w[:m] ** 2))
            cc = (4 + mueff / (d + 4)) / ((d + 2) + 2 * mueff / (d + 4))
            cs = (mueff + 2) / (d + mueff + 5)
            c1 = 2 / ((d + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
            chi = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

            C = np.eye(d)
            sigma = 1.0
            ps = np.zeros(d)
            pc = np.zeros(d)

            # Run CMA-ES for this restart
            max_gen = cma_budget_per_restart // lam
            for gen in range(max_gen):
                if calls >= budget or time.time() > deadline:
                    break

                # Generate population
                xs = []
                fs = []
                for i in range(lam):
                    if calls >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(d)
                    # Use Cholesky of C
                    try:
                        B = np.linalg.cholesky(C + 1e-12 * np.eye(d))
                    except np.linalg.LinAlgError:
                        B = np.eye(d)
                    x = clip(mu + sigma * (B @ z))
                    f = float(func(x))
                    calls += 1
                    xs.append(x)
                    fs.append(f)
                    if f < best_f:
                        best_f, best_x = f, x.copy()

                if len(xs) < 2:
                    break

                # Sort
                order = np.argsort(fs)
                xs = [xs[i] for i in order[:m]]
                fs = [fs[i] for i in order[:m]]

                # Update mean
                new_mu = np.zeros(d)
                for i in range(m):
                    new_mu += w[i] * xs[i]
                mu = new_mu

                # Update path
                for i in range(m):
                    diff = (xs[i] - mu) / sigma
                    ps = (1 - cs) * ps + cs * diff
                    pc = (1 - cc) * pc + cc * (ps @ C) / (chi ** 2)

                # Update covariance
                C = (1 + c1 + cmu - c1) * C
                C += c1 * (np.outer(ps, ps) - C)
                for i in range(m):
                    diff = (xs[i] - mu) / sigma
                    C += cmu * w[i] * (np.outer(diff, diff) - C)

                # Symmetrize
                C = (C + C.T) / 2.0

                # Update sigma (CMS)
                sigma *= np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1))
                sigma = min(sigma, 10.0)
                mu = clip(mu)

        return best_x, best_f
# EVOLVE-BLOCK-END
