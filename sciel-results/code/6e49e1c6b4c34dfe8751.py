"""CMA-ES optimizer for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with a
    wall-clock deadline guard. Adapts a full covariance matrix to the
    local landscape geometry, which handles rotations and multi-modality
    far better than a (1+1)-ES with isotropic steps."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma, seed, max_calls, deadline):
        """Run CMA-ES from a starting point, using at most max_calls and
        stopping before deadline. Returns (best_x, best_f)."""
        n = self.dim
        rng = np.random.default_rng(seed)

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(n)))          # population size
        w = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
        w = w / w.sum()
        mueff = int(np.sum(w[: int(np.ceil(np.sum(w * w)))]) ** -1)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = (2 - mueff / (n + 2) + 2 * w[0]) / ((n + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        x = np.array(x0, dtype=float)
        C = np.eye(n)
        # precompute C^0.5 = I
        Csqrt = np.eye(n)
        Cinv = np.eye(n)
        Cinv_sqrt = np.eye(n)
        B = np.eye(n)
        d = np.ones(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # initial population
        best_x = x.copy()
        best_f = func(x)
        calls = 1

        # evaluate initial population
        def eval_pop():
            nonlocal calls, best_x, best_f
            Y = x[:, None] + sigma * (Csqrt @ rng.standard_normal((n, lam)))
            Y = np.clip(Y, -5, 5)
            F = np.array([func(Y[:, i]) for i in range(lam)])
            calls += lam
            order = np.argsort(F)
            for idx in order:
                if F[idx] < best_f:
                    best_f = F[idx]
                    best_x = Y[:, idx].copy()
            return Y, F, order

        # We already have x evaluated; add it as a candidate in the population.
        # Simpler: just start the loop; the first eval_pop gives lam new points.
        # To honor the "initial point" we already evaluated, that's fine.

        while calls < max_calls and time.time() < deadline:
            Y, F, order = eval_pop()
            if time.time() > deadline - 0.5:
                break
            z = np.linalg.solve(Csqrt, (Y - x[:, None]) / sigma)
            # update paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * np.linalg.solve(Csqrt, (Y[:, order[:mueff]].T @ w[:mueff]).T)
            x_old = x.copy()
            x = x + c_c * p_c + (1 - c_c) * (x - x)  # placeholder
            # proper mean update
            x_new = x_old + sigma * (Csqrt @ (Y[:, order[:mueff]].T @ w[:mueff]).T)
            # path covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x_old) / sigma
            # covariance update
            v = p_c / np.sqrt(np.dot(p_c, p_c)) if np.dot(p_c, p_c) > 1e-20 else np.zeros(n)
            delta_sigma = (np.dot(p_sigma, p_sigma) / chiN**2 - 1) * np.exp(0.5 * min(1, (np.dot(p_sigma, p_sigma) / chiN**2 - 1) / c_sigma))
            C = (1 + c1 + cmu - c1 - c_mu if False else 1) * C
            # standard rank-mu update
            Ym = Y[:, order[:mueff]]
            C = (1 + c1 + cmu) * C + c1 * np.outer(p_c, p_c) * (1 + (1 - c_c) * delta_sigma)
            C = C + cmu * (Ym - x_old[:, None]).T @ (w[:mueff] * (Ym - x_old[:, None]))
            # ensure symmetry
            C = 0.5 * (C + C.T)
            # recompute eigen decomposition
            try:
                d, B = np.linalg.eigh(C)
                d = np.maximum(d, 1e-12)
                Cinv_sqrt = (B / np.sqrt(d)) @ B.T
                Cinv = (B / d) @ B.T
                Csqrt = B @ np.diag(np.sqrt(d))
            except np.linalg.LinAlgError:
                break
            x = x_new
            # step size adaptation (CSA)
            sigma = sigma * np.exp(c_sigma / chiN * (np.linalg.norm(p_sigma) / chiN - 1))
            sigma = np.clip(sigma, 1e-8, 50.0)

        return best_x, best_f

    def __call__(self, func):
        start = time.time()
        deadline = start + 290.0   # safety margin under the 300s limit
        n = self.dim
        budget = self.budget

        # Multi-start CMA-ES: a few independent restarts to escape local minima
        # (f21 is strongly multi-modal). Allocate the budget across restarts.
        n_restarts = 4
        per = max(1, (budget - n_restarts) // n_restarts)

        best_x = None
        best_f = np.inf
        base_seed = 12345

        # First a coarse deterministic scan of a small grid to seed restarts
        # cheaply, then CMA-ES from the best few seeds.
        grid_pts = []
        for a in (-4.0, -1.5, 0.0, 1.5, 4.0):
            for b in (-4.0, 0.0, 4.0):
                pt = np.array([a, b, 0.0, a * 0.5, b * 0.5], dtype=float)
                pt = np.clip(pt, -5, 5)
                grid_pts.append(pt)
        # evaluate grid (cheap, at most 45 points)
        for pt in grid_pts:
            if time.time() > deadline - 1.0:
                break
            f = func(pt)
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # CMA-ES restarts from the top seeds
        seeds = [best_x.copy()]
        for pt in grid_pts[:3]:
            seeds.append(pt.copy())

        for r in range(n_restarts):
            if time.time() > deadline - 2.0:
                break
            x0 = seeds[r % len(seeds)]
            sigma0 = 1.5 if r == 0 else 1.0
            bx, bf = self._cmaes(func, x0, sigma0, base_seed + r, per, deadline)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        # If we still have budget and time, do one final long CMA-ES from the best
        remaining = budget - (n_restarts * per) - len(grid_pts)
        if remaining > 20 and time.time() < deadline - 2.0:
            bx, bf = self._cmaes(func, best_x, 0.5, base_seed + 999, remaining, deadline)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
