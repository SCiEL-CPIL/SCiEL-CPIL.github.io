"""Multi-objective optimizer for RE37: NSGA-II with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(100, max(40, budget // 2))
        # Crossover rate, mutation rate
        cx_rate = 0.9
        mut_rate = 0.1

        rng = np.random.default_rng(42)

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Keep an archive of all evaluated points
        archive_X = [X.copy()]
        archive_F = [F.copy()]

        def fast_non_dominated_sort(F_arr):
            """Return list of fronts (lists of indices)."""
            n = len(F_arr)
            S = [[] for _ in range(n)]  # domination set
            n_dom = [0] * n  # domination count
            fronts = [[]]

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
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
                fronts.append(next_front)
            fronts.pop()  # remove last empty
            return fronts

        def crowding_distance(F_arr, indices):
            """Return crowding distances for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F_arr[indices, m]
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                val_range = vals[order[-1]] - vals[order[0]]
                if val_range > 1e-12:
                    for i in range(1, n - 1):
                        dist[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / val_range
            return dist

        def tournament_select(fronts, F_arr, pop_size, rng):
            """Select pop_size individuals via tournament selection."""
            # Build rank array
            rank = np.zeros(len(F_arr), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i

            selected = []
            while len(selected) < pop_size:
                # Tournament of 2
                i = rng.integers(len(F_arr))
                j = rng.integers(len(F_arr))
                if rank[i] < rank[j]:
                    selected.append(i)
                elif rank[j] < rank[i]:
                    selected.append(j)
                else:
                    # Same rank, use crowding distance
                    front_idx = rank[i]
                    front = fronts[front_idx]
                    cd = crowding_distance(F_arr, front)
                    # Find positions of i and j in front
                    pos_i = list(front).index(i)
                    pos_j = list(front).index(j)
                    if cd[pos_i] > cd[pos_j]:
                        selected.append(i)
                    else:
                        selected.append(j)
            return selected

        def crossover(x1, x2, rng):
            if rng.random() > cx_rate:
                return x1.copy(), x2.copy()
            alpha = rng.random()
            c1 = alpha * x1 + (1 - alpha) * x2
            c2 = (1 - alpha) * x1 + alpha * x2
            return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

        def mutate(x, rng):
            mask = rng.random(dim) < mut_rate
            x_new = x.copy()
            x_new[mask] = lo[mask] + (hi[mask] - lo[mask]) * rng.random(np.sum(mask))
            return np.clip(x_new, lo, hi)

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Sort current population
            fronts = fast_non_dominated_sort(F)
            parents = tournament_select(fronts, F, pop_size, rng)

            # Create children
            child_X = []
            child_F = []
            for _ in range(pop_size // 2):
                i, j = parents[:2]
                parents = parents[2:] + parents[:2]  # rotate
                c1, c2 = crossover(X[i], X[j], rng)
                c1 = mutate(c1, rng)
                c2 = mutate(c2, rng)
                child_X.append(c1)
                child_F.append(problem(c1))
                used += 1
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                child_X.append(c2)
                child_F.append(problem(c2))
                used += 1
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

            if not child_X:
                break

            child_X = np.array(child_X)
            child_F = np.array(child_F)

            # Combine parent and child
            combined_X = np.vstack([X, child_X])
            combined_F = np.vstack([F, child_F])

            # Sort combined
            fronts = fast_non_dominated_sort(combined_F)

            # Select next population
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to select from this front
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(combined_F, front)
                    # Sort by crowding distance descending
                    order = np.argsort(-cd)
                    for pos in order[:remaining]:
                        idx = front[pos]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

            # Update archive
            archive_X.append(X.copy())
            archive_F.append(F.copy())

        return X
# EVOLVE-BLOCK-END
