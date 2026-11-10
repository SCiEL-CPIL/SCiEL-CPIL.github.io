# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ1."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: balance exploration and exploitation
        pop_size = min(50, max(20, budget // 4))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Returns list of fronts, each front is list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_doms = np.zeros(n, dtype=int)  # number of points dominating i
            fronts = []
            front0 = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_doms[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_doms[i] += 1
                if n_doms[i] == 0:
                    front0.append(i)

            fronts.append(front0)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            next_front.append(j)
                if next_front:
                    k += 1
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(n_obj):
                f_vals = F[indices, m]
                f_min = f_vals.min()
                f_max = f_vals.max()
                if f_max - f_min < 1e-12:
                    continue
                sorted_idx = np.argsort(f_vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (f_vals[sorted_idx[i + 1]] - f_vals[sorted_idx[i - 1]]) / (f_max - f_min)
            return dist

        def tournament_select(F, fronts, k=2):
            """Tournament selection based on rank and crowding distance."""
            n = len(F)
            # Compute ranks
            ranks = np.zeros(n, dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            # Compute crowding distances per front
            crowd = np.zeros(n)
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        crowd[idx] = cd[i]

            # Tournament
            while True:
                a = rng.integers(n)
                b = rng.integers(n)
                if ranks[a] < ranks[b]:
                    return a
                elif ranks[b] < ranks[a]:
                    return b
                else:
                    if crowd[a] > crowd[b]:
                        return a
                    else:
                        return b

        # Main evolution loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Fast non-dominated sort
            fronts = fast_non_dominated_sort(F)

            # Generate offspring
            offspring_X = np.zeros((pop_size, dim))
            offspring_F = np.zeros((pop_size, n_obj))

            for i in range(pop_size):
                if time.time() > deadline:
                    break

                # Select parents
                p1 = tournament_select(F, fronts)
                p2 = tournament_select(F, fronts)

                # SBX crossover
                child = X[p1].copy()
                for d in range(dim):
                    if rng.random() < 0.5:
                        # Polynomial distribution
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / 21)
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / 21)
                        x1, x2 = X[p1, d], X[p2, d]
                        if abs(x1 - x2) < 1e-14:
                            child[d] = x1
                        else:
                            if rng.random() < 0.5:
                                child[d] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                            else:
                                child[d] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                    else:
                        child[d] = X[p1, d]

                # Mutation
                for d in range(dim):
                    if rng.random() < 1.0 / dim:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / 21)
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / 21)
                        if u <= 0.5:
                            child[d] = child[d] + beta * (hi[d] - child[d])
                        else:
                            child[d] = child[d] - beta * (child[d] - lo[d])

                child = np.clip(child, lo, hi)
                offspring_X[i] = child
                offspring_F[i] = problem(child)
                used += 1

                if used >= budget:
                    break

            if used >= budget:
                break

            # Combine parent and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])

            # Non-dominated sort on combined population
            fronts_all = fast_non_dominated_sort(F_all)

            # Select next generation
            new_X = np.zeros((pop_size, dim))
            new_F = np.zeros((pop_size, n_obj))
            count = 0
            for front in fronts_all:
                if count + len(front) <= pop_size:
                    for idx in front:
                        new_X[count] = X_all[idx]
                        new_F[count] = F_all[idx]
                        count += 1
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - count
                    if remaining <= 0:
                        break
                    cd = crowding_distance(F_all, front)
                    sorted_front = np.argsort(-cd)
                    for i in range(min(remaining, len(sorted_front))):
                        idx = front[sorted_front[i]]
                        new_X[count] = X_all[idx]
                        new_F[count] = F_all[idx]
                        count += 1
                    break
                if count >= pop_size:
                    break

            X = new_X
            F = new_F

        return X
# EVOLVE-BLOCK-END
