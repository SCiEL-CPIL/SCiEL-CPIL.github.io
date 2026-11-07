"""Multi-objective optimizer for WFG4 using NSGA-II with adaptive mutation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective evolutionary optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        pop_size = min(100, max(20, budget // 2))
        # Ensure pop_size is even for crossover
        if pop_size % 2 != 0:
            pop_size += 1

        # Initial population
        rng = np.random.default_rng(42)
        X = lo + rng.random((pop_size, dim)) * span
        F = np.array([problem(x) for x in X], dtype=np.float64)
        evals = pop_size

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            nd = np.zeros(n, dtype=int)  # nd[i] = number of points dominating i
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        nd[j] += 1
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        nd[i] += 1

            for i in range(n):
                if nd[i] == 0:
                    fronts[0].append(i)

            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            next_front.append(q)
                i += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)

            distances = np.zeros(len(indices), dtype=np.float64)
            m = len(indices)
            F_sub = F[indices]

            for k in range(n_obj):
                # Sort by objective k
                order = np.argsort(F_sub[:, k])
                sorted_indices = np.array(indices)[order]
                distances[order[0]] = np.inf
                distances[order[-1]] = np.inf

                min_val = F_sub[order[0], k]
                max_val = F_sub[order[-1], k]
                range_val = max_val - min_val
                if range_val == 0:
                    continue

                for i in range(1, m - 1):
                    idx = sorted_indices[i]
                    pos_in_distances = list(indices).index(idx)
                    distances[pos_in_distances] += (
                        F_sub[order[i + 1], k] - F_sub[order[i - 1], k]
                    ) / range_val

            return distances

        def tournament_select(F, fronts, crowding, k=2):
            """Tournament selection based on non-dominated rank and crowding distance."""
            n = len(F)
            selected = []
            for _ in range(k):
                i = rng.integers(n)
                selected.append(i)

            # Find the best among selected
            best = selected[0]
            best_rank = fronts.index([i for i in fronts if best in i])
            for i in selected[1:]:
                rank = fronts.index([i for i in fronts if i in i])
                if rank < best_rank:
                    best = i
                    best_rank = rank
                elif rank == best_rank:
                    # Compare crowding distance
                    cd_best = crowding[fronts[best_rank].index(best)]
                    cd_i = crowding[fronts[rank].index(i)]
                    if cd_i > cd_best:
                        best = i
            return best

        def binary_crossover(x1, x2, cx_rate=0.9):
            """BLX-alpha style crossover."""
            if rng.random() > cx_rate:
                return x1.copy(), x2.copy()
            alpha = 0.5
            child1 = np.zeros(dim)
            child2 = np.zeros(dim)
            for j in range(dim):
                if x1[j] < x2[j]:
                    child1[j] = np.random.uniform(x1[j] - alpha * (x2[j] - x1[j]),
                                                   x2[j] + alpha * (x2[j] - x1[j]))
                    child2[j] = np.random.uniform(x1[j] - alpha * (x2[j] - x1[j]),
                                                   x2[j] + alpha * (x2[j] - x1[j]))
                else:
                    child1[j] = np.random.uniform(x2[j] - alpha * (x1[j] - x2[j]),
                                                   x1[j] + alpha * (x1[j] - x2[j]))
                    child2[j] = np.random.uniform(x2[j] - alpha * (x1[j] - x2[j]),
                                                   x1[j] + alpha * (x1[j] - x2[j]))
            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)
            return child1, child2

        def mutate(x, rate=0.1):
            """Polynomial-like mutation."""
            if rng.random() > rate:
                return x.copy()
            # Use Gaussian mutation with adaptive step
            step = 0.1 * span
            x_mut = x + rng.standard_normal(dim) * step
            return np.clip(x_mut, lo, hi)

        # Main loop
        while evals < budget:
            # Check time
            if time.time() > deadline:
                break

            # Compute fronts and crowding
            fronts = fast_non_dominated_sort(F)
            crowding = np.zeros(pop_size, dtype=np.float64)
            for front in fronts:
                if len(front) > 2:
                    cd = crowding_distance(F, front)
                    for idx, val in zip(front, cd):
                        crowding[idx] = val
                else:
                    for idx in front:
                        crowding[idx] = np.inf

            # Generate offspring
            offspring_X = []
            offspring_F = []

            while len(offspring_X) < pop_size:
                # Check time
                if time.time() > deadline:
                    break

                # Select parents
                p1 = tournament_select(F, fronts, crowding)
                p2 = tournament_select(F, fronts, crowding)

                # Crossover
                c1, c2 = binary_crossover(X[p1], X[p2])

                # Mutation
                c1 = mutate(c1)
                c2 = mutate(c2)

                # Evaluate
                f1 = problem(c1)
                evals += 1
                offspring_X.append(c1)
                offspring_F.append(f1)

                if evals >= budget:
                    break

                if time.time() > deadline:
                    break

                f2 = problem(c2)
                evals += 1
                offspring_X.append(c2)
                offspring_F.append(f2)

            if not offspring_X:
                break

            # Combine parents and offspring
            X_all = np.vstack([X, np.array(offspring_X)])
            F_all = np.vstack([F, np.array(offspring_F)])

            # Selection: keep best pop_size from combined
            fronts_all = fast_non_dominated_sort(F_all)
            new_X = []
            new_F = []

            for front in fronts_all:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_all[idx])
                        new_F.append(F_all[idx])
                else:
                    # Need to select from this front based on crowding
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(F_all, front)
                    order = np.argsort(cd)[::-1]
                    for i in range(min(remaining, len(order))):
                        idx = front[order[i]]
                        new_X.append(X_all[idx])
                        new_F.append(F_all[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

            # Adaptive mutation rate based on stagnation
            # (could be enhanced, but keeping simple for now)

        return X
# EVOLVE-BLOCK-END
