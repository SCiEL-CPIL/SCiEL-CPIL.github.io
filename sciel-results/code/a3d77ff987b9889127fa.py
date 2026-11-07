"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT1 using NSGA-II style search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Population size
        pop_size = min(100, max(20, budget // 3))

        # Initialize population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Return list of non-dominated fronts."""
            n = len(F)
            S = [set() for _ in range(n)]  # Dominated sets
            nd = [0] * n  # Dominance count
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        nd[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        nd[i] += 1

            for i in range(n):
                if nd[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)

            return fronts[:-1]  # Remove last empty front

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n

            distances = np.zeros(n)
            m = len(front_indices[0])  # number of objectives

            for k in range(m):
                obj_vals = np.array([F[i][k] for i in front_indices])
                sorted_idx = np.argsort(obj_vals)
                distances[sorted_idx[0]] = float('inf')
                distances[sorted_idx[-1]] = float('inf')
                obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                if obj_range == 0:
                    continue
                for i in range(1, n - 1):
                    distances[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / obj_range

            return distances

        def tournament_select(F, fronts, k=2):
            """Tournament selection based on non-domination rank and crowding distance."""
            # Compute ranks
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i

            # Compute crowding distances
            cd = np.zeros(len(F))
            for front in fronts:
                if front:
                    cd_list = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd[idx] = cd_list[i]

            # Tournament
            candidates = rng.integers(0, len(F), size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best]:
                    best = c
                elif rank[c] == rank[best] and cd[c] > cd[best]:
                    best = c
            return best

        def crossover(x1, x2, rate=0.9):
            """SBX crossover."""
            if rng.random() > rate:
                return x1.copy(), x2.copy()

            child1 = x1.copy()
            child2 = x2.copy()

            for i in range(len(x1)):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (2 + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (2 + 1))
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])

            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        def mutate(x, rate=0.1):
            """Polynomial mutation."""
            x_mut = x.copy()
            for i in range(len(x_mut)):
                if rng.random() < rate:
                    u = rng.random()
                    delta = hi[i] - lo[i]
                    if u < 0.5:
                        x_mut[i] = x_mut[i] + delta * (2 * u) ** 6 * (rng.random() - 0.5)
                    else:
                        x_mut[i] = x_mut[i] + delta * (2 * (1 - u)) ** 6 * (rng.random() - 0.5)
            return np.clip(x_mut, lo, hi)

        # Main loop
        while used < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break

            fronts = fast_non_dominated_sort(F)
            if not fronts or not fronts[0]:
                break

            # Create offspring
            offspring_X = []
            offspring_F = []

            while len(offspring_X) < pop_size and used < budget:
                p1 = tournament_select(F, fronts)
                p2 = tournament_select(F, fronts)

                c1, c2 = crossover(X[p1], X[p2])
                c1 = mutate(c1)
                c2 = mutate(c2)

                f1 = problem(c1)
                f2 = problem(c2)
                used += 2

                if used > budget:
                    break

                offspring_X.append(c1)
                offspring_F.append(f1)

                if used < budget:
                    offspring_X.append(c2)
                    offspring_F.append(f2)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X[:pop_size])
            offspring_F = np.array(offspring_F[:pop_size])

            # Combine and select
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            combined_fronts = fast_non_dominated_sort(combined_F)

            # Select next population
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to truncate this front
                    needed = pop_size - len(new_X)
                    if needed <= 0:
                        break
                    # Sort by crowding distance
                    cd_list = crowding_distance(combined_F, front)
                    sorted_front = sorted(range(len(front)), key=lambda i: cd_list[i], reverse=True)
                    for i in range(min(needed, len(sorted_front))):
                        idx = front[sorted_front[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X[:pop_size])
            F = np.array(new_F[:pop_size])

        return X
# EVOLVE-BLOCK-END
