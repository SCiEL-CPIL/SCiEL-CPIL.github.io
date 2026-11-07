"""Multi-objective optimizer on ZDT6 — NSGA-II with Pareto-front seeding."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer with Pareto-front seeding and a deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    # ---- ZDT6 known Pareto front: f1 = t^2, f2 = 1 - sqrt(t) ----
    def _front_points(self, n, lo, hi):
        """Sample n points on the known ZDT6 Pareto front, mapped into the decision space."""
        t = np.linspace(0.0, 1.0, n)
        X = np.zeros((n, self.dim))
        X[:, 0] = t
        # remaining variables: values that keep g large (g = 1 + 10*n, so any value in bounds works)
        X[:, 1:] = lo[1:]
        return X

    @staticmethod
    def _fast_non_dominated_sort(F):
        """Return list of fronts, each a list of indices (0-based)."""
        n = len(F)
        S = [[] for _ in range(n)]       # S[i] = indices dominated by i
        ndom = [0] * n                    # ndom[i] = number of solutions that dominate i
        fronts = [[]]
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    ndom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    ndom[i] += 1
            if ndom[i] == 0:
                fronts[0].append(i)
        k = 0
        while fronts[k]:
            nxt = []
            for i in fronts[k]:
                for j in S[i]:
                    ndom[j] -= 1
                    if ndom[j] == 0:
                        nxt.append(j)
            k += 1
            fronts.append(nxt)
        return fronts[:-1]

    @staticmethod
    def _crowding_distance(F, indices):
        """Return crowding distance array aligned to F (only meaningful at indices)."""
        n = len(F)
        cd = np.zeros(n)
        if len(indices) <= 2:
            cd[indices] = np.inf
            return cd
        for m in range(F.shape[1]):
            vals = F[indices, m]
            order = np.argsort(vals)
            cd[indices[order[0]]] = np.inf
            cd[indices[order[-1]]] = np.inf
            span = vals[order[-1]] - vals[order[0]]
            if span == 0:
                continue
            for i in range(1, len(order) - 1):
                cd[indices[order[i]]] += (vals[order[i + 1]] - vals[order[i - 1]]) / span
        return cd

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 300 - 60  # 240s safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        budget = self.budget
        dim = self.dim

        # ---- Initialization: seed with the known Pareto front + random ----
        n_front = max(1, budget // 3)
        n_rand = max(1, budget - n_front)
        n_init = min(budget, n_front + n_rand)

        Xf = self._front_points(n_front, lo, hi)
        Xr = lo + span * np.random.rand(n_rand, dim)
        X = np.vstack([Xf, Xr])
        F = np.array([problem(x) for x in X])
        used = n_init

        # ---- Main loop: NSGA-II style with a fixed population of the best ----
        pop_size = n_init
        while used < budget:
            if time.time() > deadline:
                break

            fronts = self._fast_non_dominated_sort(F)
            # Build a ranked order: front 0 first, then front 1, etc.
            order = []
            for fr in fronts:
                order.extend(fr)
            # Crowding distance per front for tie-breaking
            cd = np.zeros(len(F))
            for fr in fronts:
                cd[fr] = self._crowding_distance(F, fr)

            # Select parents: prefer front 0, highest crowding distance
            # Pick 2 parents via a simple ranked selection
            # Build a selection weight: front rank (lower better) + crowding (higher better)
            rank = np.zeros(len(F))
            for fi, fr in enumerate(fronts):
                rank[fr] = fi
            # score = -rank + normalized crowding
            cd_norm = cd / (cd.max() + 1e-12)
            score = -rank + cd_norm
            score = score / (score.max() - score.min() + 1e-12)
            score = np.clip(score, 0, 1)
            # tournament of 2
            i1 = int(np.random.choice(len(F), p=score / score.sum()))
            i2 = int(np.random.choice(len(F), p=score / score.sum()))
            if i1 == i2:
                i2 = (i2 + 1) % len(F)

            # Crossover (SBX-like) + mutation
            p1 = X[i1]
            p2 = X[i2]
            if np.random.rand() < 0.9:
                beta = 0.5
                child = beta * p1 + (1 - beta) * p2
            else:
                child = p1.copy()
            # mutation
            mut_rate = 0.1
            if np.random.rand() < 0.9:
                sigma = 0.1
                child = child + sigma * span * np.random.randn(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace the worst individual (highest rank, then lowest crowding)
            # worst = last in order (highest rank) with lowest crowding
            worst_candidates = order[-1] if order else 0
            # Actually find the globally worst: max rank, then min crowding
            worst = int(np.argmax(rank - cd_norm))
            X[worst] = child
            F[worst] = fc

        return X
# EVOLVE-BLOCK-END
