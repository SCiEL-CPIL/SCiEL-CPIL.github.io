"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-inspired multi-objective optimizer with hypervolume-guided replacement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # 300s limit, 20s safety margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Initial population
        pop_size = min(40, self.budget // 5)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]
            nd = [0] * n
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] += 1
            front0 = [i for i in range(n) if nd[i] == 0]
            if front0:
                fronts.append(front0)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for indices in a front."""
            n = len(front_indices)
            if n <= 2:
                return [1e10] * n
            dist = [0.0] * n
            for m in range(self.n_obj):
                vals = [F[i][m] for i in front_indices]
                order = np.argsort(vals)
                dist[order[0]] = 1e10
                dist[order[-1]] = 1e10
                if vals[order[-1]] - vals[order[0]] > 1e-12:
                    for k in range(1, n - 1):
                        dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (vals[order[-1]] - vals[order[0]])
            return dist

        # Main loop
        while used < self.budget:
            if time.time() > deadline - 5.0:
                break

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Select parents from best front(s)
            parent_pool = []
            for front in fronts:
                if len(parent_pool) + len(front) >= max(2, len(F) // 2):
                    parent_pool.extend(front)
                    break
                parent_pool.extend(front)
            if not parent_pool:
                parent_pool = list(range(len(F)))

            # Generate children via mutation + crossover
            n_children = min(10, self.budget - used)
            children_X = []
            children_F = []
            for _ in range(n_children):
                # Pick two parents
                p1_idx = parent_pool[rng.integers(len(parent_pool))]
                p2_idx = parent_pool[rng.integers(len(parent_pool))]
                p1 = X[p1_idx]
                p2 = X[p2_idx]

                # SBX-like crossover
                if rng.random() < 0.8:
                    beta = 0.5
                    child1 = beta * p1 + (1 - beta) * p2
                    child2 = beta * p2 + (1 - beta) * p1
                else:
                    child1 = p1.copy()
                    child2 = p2.copy()

                # Mutation
                for child in [child1, child2]:
                    if rng.random() < 0.5:
                        child = child + 0.1 * (hi - lo) * rng.standard_normal(self.dim)
                    child = np.clip(child, lo, hi)

                fc1 = problem(child1)
                used += 1
                children_X.append(child1)
                children_F.append(fc1)

                if used < self.budget:
                    fc2 = problem(child2)
                    used += 1
                    children_X.append(child2)
                    children_F.append(fc2)

                if used >= self.budget:
                    break

            # Combine and select
            if not children_X:
                break

            X_all = np.vstack([X, np.array(children_X)])
            F_all = np.vstack([F, np.array(children_F)])

            # Select best pop_size from combined
            fronts_all = fast_non_dominated_sort(F_all)
            selected = []
            for front in fronts_all:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    # Need to trim this front
                    remaining = pop_size - len(selected)
                    dist = crowding_distance(F_all, front)
                    order = np.argsort(-dist)
                    selected.extend(front[i] for i in order[:remaining])
                    break

            X = X_all[selected]
            F = F_all[selected]

        return X
# EVOLVE-BLOCK-END
