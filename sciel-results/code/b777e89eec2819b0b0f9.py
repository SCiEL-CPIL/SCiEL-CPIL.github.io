"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT3: structured init, crowding tournament,
    vectorized dominance, time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before the 330s wall-clock kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        def non_dominated(F):
            """Boolean mask of non-dominated rows in F (n x n_obj)."""
            n = len(F)
            if n == 0:
                return np.array([], dtype=bool)
            # point i is dominated by j iff F[j] <= F[i] elementwise and F[j] != F[i]
            leq = F[:, np.newaxis, :] <= F[np.newaxis, :, :]
            neq = F[:, np.newaxis, :] != F[np.newaxis, :, :]
            dominated = np.any(np.all(leq, axis=2) & np.any(neq, axis=2), axis=1)
            return ~dominated

        def crowding_distance(F):
            n = len(F)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for k in range(n_obj):
                order = np.argsort(F[:, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                f_range = F[order[-1], k] - F[order[0], k]
                if f_range > 0:
                    for i in range(1, n - 1):
                        cd[order[i]] += (F[order[i + 1], k] - F[order[i - 1], k]) / f_range
            return cd

        pop_size = min(100, max(20, self.budget // 2))

        # ---- Initial population: mix of random and ZDT3-structured points ----
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        # Structure ~25% of the population near the ZDT3 front:
        #   x1 = 0  (first decision var drives the discontinuous front),
        #   g = 1   (minimal aggregation -> points on the front),
        #   x2..xM  spread across [0,1] to cover the multimodal front.
        n_struct = max(1, pop_size // 4)
        for s in range(n_struct):
            x = np.zeros(dim)
            x[0] = 0.0
            # g(x) = 1 + 10*(M-1) at x1=0; set g to 1 => all remaining vars = 0
            x[1:] = 0.0
            # spread a few decision vars to explore the front's sub-fronts
            spread = rng.random(dim - 1)
            x[1:] = spread
            X[s] = np.clip(x, lo, hi)

        F = np.array([problem(x) for x in X])
        used = pop_size

        while used < self.budget:
            if time.time() > deadline:
                break

            mask = non_dominated(F)
            nd_idx = np.where(mask)[0]
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))
            parents = X[nd_idx]
            F_nd = F[nd_idx]

            # Crowding-distance tournament selection among the non-dominated set
            if len(parents) > 1:
                cd = crowding_distance(F_nd)
                t1 = rng.integers(len(parents))
                t2 = rng.integers(len(parents))
                parent_idx = t1 if cd[t1] >= cd[t2] else t2
            else:
                parent_idx = 0

            parent = parents[parent_idx]
            child = parent + 0.15 * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc

        return X
# EVOLVE-BLOCK-END
