# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Grid-based multi-objective optimizer: Sobol coarse map + Nelder-Mead refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim, n_obj = self.dim, self.n_obj

        # Phase 1: Coarse Sobol grid for systematic coverage
        n_grid = min(100, self.budget // 3)
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            X_grid = sampler.random(n_grid)
        except Exception:
            rng = np.random.default_rng(42)
            X_grid = rng.random((n_grid, dim))
        X_grid = lo + span * X_grid
        F_grid = np.array([problem(x) for x in X_grid])
        used = n_grid

        # Helper: non-dominated filter
        def non_dominated_indices(F):
            n = len(F)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] = False
                        break
            return np.where(nd)[0]

        nd_idx = non_dominated_indices(F_grid)
        candidates = F_grid[nd_idx]

        # Sort candidates by sum of objectives (lower = more central on front)
        if len(candidates) > 0:
            order = np.argsort(candidates.sum(axis=1))
            nd_idx = nd_idx[order]

        # Phase 2: Nelder-Mead refinement on top non-dominated grid points
        # Allocate remaining budget: ~60 evals per refinement
        evals_per_refine = 60
        max_refines = max(1, (self.budget - used) // evals_per_refine)
        n_refine = min(len(nd_idx), max_refines)

        for k in range(n_refine):
            if used >= self.budget:
                break
            if time.time() > deadline - 10.0:
                break

            x0 = X_grid[nd_idx[k]]

            # Weighted-sum scalarization: weights based on rank (spread weights)
            n_cand = len(nd_idx)
            w = np.ones(n_obj) / n_obj
            # Bias weights toward the candidate's direction
            if n_cand > 1:
                fmin = F_grid.min(axis=0)
                fmax = F_grid.max(axis=0)
                rng_norm = fmax - fmin
                rng_norm[rng_norm == 0] = 1.0
                f_norm = (F_grid - fmin) / rng_norm
                # Weight inversely proportional to distance from candidate in normalized space
                dist = np.sqrt(((f_norm - f_norm[k]) ** 2).sum(axis=1))
                w = 1.0 / (dist + 0.1)
                w = w / w.sum()

            def scalar(x):
                fx = problem(x)
                return float(np.dot(fx, w))

            try:
                res = minimize(scalar, x0, method='Nelder-Mead',
                               bounds=list(zip(lo, hi)),
                               options={'maxiter': evals_per_refine,
                                        'xatol': 1e-5, 'fatol': 1e-8})
                # Count evaluations used by this refinement
                n_evals_used = res.nfev if hasattr(res, 'nfev') else evals_per_refine
                used += min(n_evals_used, self.budget - used)
            except Exception:
                pass

        return X_grid
# EVOLVE-BLOCK-END
