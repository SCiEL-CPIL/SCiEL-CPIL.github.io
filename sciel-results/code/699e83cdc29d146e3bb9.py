"""Multi-objective optimizer on RE21 (2 objectives, 4 variables)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Hypervolume-driven evolutionary multi-objective search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # safety margin well under the 300s limit

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(20240517)  # deterministic, seed-independent

        def hv_contribution(F):
            """Sum of 2-objective hypervolume contributions (reference-free, all positive)."""
            n = len(F)
            if n == 0:
                return np.zeros(0)
            if n == 1:
                return np.array([F[0, 0] * F[0, 1]])
            order = np.lexsort((F[:, 1], F[:, 0]))  # f0 asc, then f1 asc
            F = F[order]
            f1_max = np.maximum.accumulate(F[:, 1])
            f0_prev = np.concatenate(([0.0], F[:-1, 0]))
            contrib = (F[:, 0] - f0_prev) * f1_max
            out = np.empty(n)
            out[order] = contrib
            return out

        def evaluate(x):
            return np.asarray(problem(x), dtype=float)

        # ---- Latin hypercube initialization for broad coverage ----
        n_init = int(min(max(4, budget // 10), 25))
        n_init = max(1, min(n_init, budget))
        X = np.empty((n_init, dim))
        for j in range(dim):
            perm = rng.permutation(n_init)
            u = (perm + rng.random(n_init)) / n_init
            X[:, j] = lo[j] + (hi[j] - lo[j]) * u
        F = np.array([evaluate(x) for x in X], dtype=float)
        used = n_init

        # ---- Main loop: hypervolume-guided exploitation + exploration ----
        while used < budget:
            if time.time() > deadline - 1.0:
                break

            # prune dominated points from the working population
            n = len(F)
            if n > 1:
                A = np.any(F <= F[:, None, :], axis=2) & np.any(F < F[:, None, :], axis=2)
                A = A & ~np.eye(n, dtype=bool)
                keep = ~A.any(axis=1)
            else:
                keep = np.ones(n, dtype=bool)
            X = X[keep]
            F = F[keep]
            n = len(F)

            if n == 0:
                X = np.array([lo + (hi - lo) * 0.5])
                F = np.array([evaluate(X[0])])
                used += 1
                continue

            # pick two parents via tournament on hypervolume contribution
            c = hv_contribution(F)
            i1 = int(np.argmax(c[rng.integers(n, size=2)]))
            i2 = int(np.argmax(c[rng.integers(n, size=2)]))
            p1, p2 = X[i1], X[i2]

            # crossover (SBX-like blend) + mutation
            beta = 0.5 + 0.5 * rng.random()
            child = beta * p1 + (1.0 - beta) * p2
            sigma = (hi - lo) * (0.08 + 0.12 * rng.random())
            child = child + sigma * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = evaluate(child)
            used += 1

            # keep if it adds hypervolume; otherwise replace worst contributor
            if len(F) < 1:
                X = np.array([child])
                F = np.array([fc])
                continue
            c = hv_contribution(F)
            if c.min() > 0.0:
                worst = int(np.argmin(c))
                if np.any(F[worst] <= fc) and np.any(F[worst] < fc):
                    # child dominated by worst -> still replace to keep diversity
                    pass
                X[worst], F[worst] = child, fc
            else:
                # no positive contributions; keep the lexicographically best set
                if not (np.all(F <= fc) and np.any(F < fc)):
                    worst = int(np.argmin(c))
                    X[worst], F[worst] = child, fc

        return X
# EVOLVE-BLOCK-END
