"""Optimizer for BBOB f21 (Gallagher 101 peaks), dim=5, budget=500, box [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start global exploration + CMA-ES local exploitation, with a deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0          # 15 s safety margin before the 300 s kill
        dim = self.dim
        budget = self.budget

        # ---- call tracking (single source of truth) ----
        state = {"n": 0}

        def eval_point(x):
            if state["n"] >= budget:
                return None
            state["n"] += 1
            return func(np.asarray(x, dtype=float))

        def time_ok():
            return (time.time() < deadline) and (state["n"] < budget)

        # ---- deterministic initial grid: 3^5 = 243 points ----
        levels = np.array([-4.0, 0.0, 4.0])
        grid = np.array(np.meshgrid(*([levels] * dim)), dtype=float)
        grid = grid.reshape(-1, dim)

        best_x = None
        best_f = np.inf
        evaluated = []  # (f, x)

        def consider(fv, xv):
            nonlocal best_x, best_f
            if fv < best_f:
                best_f = fv
                best_x = np.array(xv, dtype=float)

        # ---- Phase 1: deterministic global exploration (243 calls) ----
        for i in range(grid.shape[0]):
            if not time_ok():
                break
            fv = eval_point(grid[i])
            if fv is None:
                break
            evaluated.append((fv, grid[i]))
            consider(fv, grid[i])

        # ---- Phase 2: CMA-ES local exploitation around the best point ----
        # ~257 calls left. Use a small population so the CMA matrix is well
        # conditioned and we can afford a few restarts.
        lambda_pop = 6
        max_iter = 40

        def cma_es(x0, f0):
            nonlocal best_x, best_f
            # sigma: a small fraction of the box; the 101-peak function has
            # relatively narrow peaks, so we start modestly and let adaptation
            # grow it if the search is expanding.
            sigma = 1.0
            x = np.array(x0, dtype=float)
            f_best = f0
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            # adaptation rates (dim=5)
            cmu = 1.0 / (4 * dim + 4)
            ccs = (4 * dim + 2) / ((dim + 4) **2 + 4)
            cc = (5 + dim) / (dim + 9)
            cs = (1 + 2 * max(0.0, (np.sqrt(dim + 1.3) - 1) / (dim + 4))) * cc
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            for _ in range(max_iter):
                if not time_ok():
                    break
                # generate lambda_pop offspring
                offspring = []
                for _ in range(lambda_pop):
                    if not time_ok():
                        break
                    z = np.random.randn(dim)
                    y = x + sigma * (z @ C)
                    y = np.clip(y, -5.0, 5.0)
                    fv = eval_point(y)
                    if fv is None:
                        return
                    offspring.append((fv, y))
                if len(offspring) < 2:
                    return
                offspring.sort(key=lambda t: t[0])
                mu = len(offspring) // 2
                # update best
                if offspring[0][0] < f_best:
                    f_best = offspring[0][0]
                    consider(f_best, offspring[0][1])
                # recombination weights (linear, normalized)
                w = np.array([np.log(mu + 0.5) - np.log(i + 0.5) for i in range(mu)])
                w = w / w.sum()
                mueff = 1.0 / np.sum(w ** 2)
                # weighted mean
                y_mean = sum(w[i] * offspring[i][1] for i in range(mu))
                # evolution path
                ps = (1 - cmu) * ps + np.sqrt(cmu * (2 - cmu) * mueff) * (C @ ((y_mean - x) / sigma))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cmu) ** (2 * (max_iter + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
                pc = (1 - ccs) * pc + hsig * np.sqrt(ccs * (2 - ccs) * cc) * (y_mean - x) / sigma
                # update covariance
                delta_d = (1 + 2 * max(0.0, (np.sqrt(mueff) - 1) / (dim + 2))) * ccs
                C = ((1 - cmu * (1 + delta_d)) * C
                     + cmu * np.outer(y_mean - x, y_mean - x) / (sigma ** 2)
                     + cmu * delta_d * np.outer(pc, pc))
                # normalize (keep unit trace)
                C = C / np.trace(C) * dim
                x = y_mean
                # step-size adaptation
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = min(max(sigma, 1e-6), 10.0)
            return f_best

        # restarts: top few distinct points from phase 1, each with its own CMA
        evaluated.sort(key=lambda t: t[0])
        # pick up to 3 well-separated starting points
        starts = []
        for fv, xv in evaluated:
            if all(np.linalg.norm(xv - s[1]) > 1.0 for s in starts):
                starts.append((fv, xv))
            if len(starts) >= 3:
                break
        if not starts:
            starts = [(best_f, best_x)]

        # distribute remaining calls across restarts
        remaining = budget - state["n"]
        per = max(2, remaining // len(starts))
        for fv, xv in starts:
            if not time_ok():
                break
            cma_es(xv, fv)

        return best_x, best_f
# EVOLVE-BLOCK-END
