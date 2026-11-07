"""Multi-objective optimizer on WFG7: NSGA-II with wall-clock deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II multi-objective optimizer with a wall-clock deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 295.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(60, budget // 2)
        if pop_size < 10:
            pop_size = max(10, budget)

        # Initial population
        rng = np.random.default_rng(42)
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of solutions dominating i
            fronts = [[]]

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def crowding_distance(F, indices):
            """Assign crowding distances to a set of indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F[indices, m]
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                val_range = vals[order[-1]] - vals[order[0]]
                if val_range > 0:
                    for i in range(1, n - 1):
                        cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / val_range
            return cd

        # Main evolutionary loop
        while used < budget:
            if time.time() > deadline:
                break

            # Evaluate offspring population
            offspring_size = pop_size
            # Crossover and mutation
            child_X = np.empty((offspring_size, dim))
            for i in range(offspring_size):
                if used + 1 > budget or time.time() > deadline:
                    break
                # Tournament selection (size 2)
                idx1 = rng.integers(0, pop_size)
                idx2 = rng.integers(0, pop_size)
                if np.all(F[idx1] <= F[idx2]) and np.any(F[idx1] < F[idx2]):
                    p1 = idx1
                else:
                    p1 = idx2

                idx3 = rng.integers(0, pop_size)
                idx4 = rng.integers(0, pop_size)
                if np.all(F[idx3] <= F[idx4]) and np.any(F[idx3] < F[idx4]):
                    p2 = idx3
                else:
                    p2 = idx4

                # Simulated binary crossover
                if rng.random() < 0.9:
                    alpha = rng.random()
                    if alpha < 0.5:
                        child_X[i] = X[p1] * alpha + X[p2] * (1 - alpha)
                    else:
                        child_X[i] = X[p1] * (1 - alpha) + X[p2] * alpha
                else:
                    child_X[i] = X[p1].copy()

                # Gaussian mutation
                mutation_rate = 1.0 / dim
                mask = rng.random(dim) < mutation_rate
                if np.any(mask):
                    child_X[i, mask] += 0.1 * span[mask] * rng.standard_normal(np.sum(mask))

                child_X[i] = np.clip(child_X[i], lo, hi)

            # Evaluate offspring
            child_F = np.array([problem(x) for x in child_X], dtype=float)
            used += len(child_X)

            # Combine parent and offspring
            all_X = np.vstack([X, child_X])
            all_F = np.vstack([F, child_F])

            # Non-dominated sort
            fronts = fast_non_dominated_sort(all_F)

            # Select next population
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(all_X[idx])
                        next_F.append(all_F[idx])
                else:
                    # Need to select some from this front using crowding distance
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(all_F, front)
                    order = np.argsort(-cd)
                    for j in range(remaining):
                        idx = front[order[j]]
                        next_X.append(all_X[idx])
                        next_F.append(all_F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
