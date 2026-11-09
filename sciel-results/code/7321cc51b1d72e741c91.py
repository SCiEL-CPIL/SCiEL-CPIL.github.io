"""Fresh multi-objective optimizer for RE21 (budget=200, dim=4, n_obj=2)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lightweight multi-objective search: stratified init + adaptive Pareto-guided mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # safety margin well under 330s wall-clock
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        mid = 0.5 * (hi + lo)

        def is_deadline():
            return (time.time() - start_time) > (deadline - start_time)

        # ---- stratified (Latin-hypercube-ish) initialization for cheap diversity ----
        init_n = min(40, max(10, budget // 5))
        rng = np.random.default_rng(0)
        X = np.empty((init_n, dim))
        for d in range(dim):
            order = rng.permutation(init_n)
            u = (order + 0.5) / init_n
            X[:, d] = lo[d] + u * span[d]
        F = np.array([problem(x) for x in X], dtype=float)
        used = init_n

        # archive of all evaluated points
        all_X = X.copy()
        all_F = F.copy()

        # adaptive step: start wide, shrink over the run
        total_steps = max(1, budget - init_n)
        step0 = 0.35  # fraction of box
        step_end = 0.02

        for it in range(total_steps):
            if used >= budget:
                break
            if is_deadline():
                break
            t = it / total_steps
            step = step0 * (step_end / step0) ** t  # geometric decay

            # choose a parent: mostly from current non-dominated set, sometimes random
            nd = self._non_dominated_idx(all_F)
            if len(nd) > 0 and rng.random() < 0.75:
                parent = all_X[nd[rng.integers(len(nd))]]
            else:
                parent = all_X[rng.integers(len(all_X))]

            # adaptive Gaussian mutation, plus occasional larger jump
            if rng.random() < 0.15:
                s = step * 2.0
            else:
                s = step
            child = parent + s * span * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            all_X = np.vstack([all_X, child])
            all_F = np.vstack([all_F, np.asarray(fc, dtype=float)])

        return all_X

    def _non_dominated_idx(self, F):
        """Return indices of non-dominated points in F (n_obj x n)."""
        n = F.shape[0]
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                # j dominates i?
                if np.all(F[:, j] <= F[:, i]) and np.any(F[:, j] < F[:, i]):
                    nd[i] = False
                    break
        return np.where(nd)[0]
# EVOLVE-BLOCK-END
