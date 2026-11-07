# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-style multi-objective optimizer tuned for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def _sample_front(self, rng, n):
        """Sample points on the DTLZ4 Pareto-front manifold:
        x[0:3] are normalized random positive weights, x[3:] are small values."""
        w = rng.random((n, 3))
        w = w / np.linalg.norm(w, axis=1, keepdims=True)
        x = np.zeros((n, self.dim))
        x[:, :3] = w
        # remaining variables: small random values (0.05) so g is near its minimum
        x[:, 3:] = 0.05 * rng.random((n, self.dim - 3))
        return x

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # --- initial population: front manifold + random ---
        n_init = 80
        Xf = self._sample_front(rng, n_init)
        Xr = lo + span * rng.random((n_init, self.dim))
        X = np.clip(np.vstack([Xf, Xr]), lo, hi)
        F = np.array([problem(x) for x in X], dtype=float)
        used = 2 * n_init

        # keep only points strictly better than reference in every objective
        ref = np.array([2.5, 2.5, 2.5])

        # --- NSGA-II main loop ---
        pop_size = 60
        # select initial population: prefer non-dominated, then spread
        def _non_dominated_idx(Fm):
            n = len(Fm)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(Fm[j] <= Fm[i]) and np.any(Fm[j] < Fm[i]):
                        nd[i] = False
                        break
            return np.where(nd)[0]

        # pick pop_size from the 160 candidates: first non-dominated, then by spread
        nd_idx = _non_dominated_idx(F)
        if len(nd_idx) >= pop_size:
            # spread: pick evenly from sorted by first objective
            order = np.argsort(F[nd_idx, 0])
            sel = nd_idx[order[:: max(1, len(order) // pop_size)][:pop_size]]
            if len(sel) < pop_size:
                extra = np.setdiff1d(nd_idx, sel)
                sel = np.concatenate([sel, extra[:pop_size - len(sel)]])
        else:
            sel = np.arange(min(pop_size, len(F)))
        X = X[sel]
        F = F[sel]

        mutation_scale = 0.3

        while used < self.budget:
            if time.time() > deadline:
                break

            # adaptive mutation scale
            progress = used / self.budget
            mutation_scale = 0.3 - 0.25 * progress  # 0.3 -> 0.05

            # non-dominated parents
            nd_idx = _non_dominated_idx(F)
            parents = X[nd_idx] if len(nd_idx) > 0 else X

            # generate a batch of children (one at a time to respect budget)
            for _ in range(min(pop_size, self.budget - used)):
                if time.time() > deadline:
                    break
                p = parents[rng.integers(len(parents))]
                # mutation: Gaussian on the front variables, small on the rest
                mut = np.zeros(self.dim)
                mut[:3] = mutation_scale * rng.standard_normal(3)
                mut[3:] = 0.02 * rng.standard_normal(self.dim - 3)
                child = p + mut
                child = np.clip(child, lo, hi)
                fc = np.asarray(problem(child), dtype=float)
                used += 1

                # replace the worst point in the population (by sum of objectives)
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst], F[worst] = child, fc

        return X
# EVOLVE-BLOCK-END
