"""Multi-objective optimizer for ZDT6 using NSGA-II."""
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Population size: use a reasonable fraction of budget
        pop_size = min(24, max(8, self.budget // 8))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dominates = [0] * n
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominates[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominates[i] += 1

                if n_dominates[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)

            return fronts[:-1]  # Remove empty last front

        def crowding_distance(F, front_indices):
            """Compute crowding distance for members of a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            for m in range(n_obj):
                obj_vals = F[front_indices, m]
                min_val = obj_vals.min()
                max_val = obj_vals.max()
                if max_val - min_val < 1e-12:
                    dist[:] = np.inf
                    continue
                # Sort by objective m
                order = np.argsort(obj_vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for i in range(1, n - 1):
                    dist[order[i]] += (obj_vals[order[i + 1]] - obj_vals[order[i - 1]]) / (max_val - min_val)

            return dist

        def tournament_select(ranks, dists, n_select=1):
            """Binary tournament selection."""
            winners = []
            for _ in range(n_select):
                i, j = rng.integers(0, len(ranks), size=2)
                if ranks[i] < ranks[j]:
                    winners.append(i)
                elif ranks[j] < ranks[i]:
                    winners.append(j)
                else:
                    if dists[i] > dists[j]:
                        winners.append(i)
                    else:
                        winners.append(j)
            return winners

        def crossover(x1, x2):
            """Simple blend crossover."""
            if rng.random() < 0.9:
                alpha = rng.random()
                child1 = alpha * x1 + (1 - alpha) * x2
                child2 = (1 - alpha) * x1 + alpha * x2
                return child1, child2
            else:
                return x1.copy(), x2.copy()

        def mutate(x):
            """Gaussian mutation."""
            sigma = 0.1
            x_mut = x + sigma * (hi - lo) * rng.standard_normal(dim)
            return np.clip(x_mut, lo, hi)

        # Main evolution loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Compute crowding distances
            dists = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    front_dists = crowding_distance(F, front)
                    for idx, d in zip(front, front_dists):
                        dists[idx] = d

            # Assign ranks
            ranks = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            # Generate offspring
            n_offspring = pop_size
            offspring_X = []
            for _ in range(n_offspring // 2):
                p1_idx = tournament_select(ranks, dists, 1)[0]
                p2_idx = tournament_select(ranks, dists, 1)[0]
                c1, c2 = crossover(X[p1_idx], X[p2_idx])
                offspring_X.append(mutate(c1))
                offspring_X.append(mutate(c2))

            offspring_X = np.array(offspring_X[:n_offspring])
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += len(offspring_F)

            if used > self.budget:
                # Trim to budget
                offspring_X = offspring_X[:self.budget - (used - len(offspring_F))]
                offspring_F = offspring_F[:self.budget - (used - len(offspring_F))]
                used = self.budget

            # Combine parent and offspring
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])

            # Non-dominated sorting on combined
            fronts = fast_non_dominated_sort(F_combined)

            # Select next generation
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                else:
                    # Need to truncate this front
                    remaining = pop_size - len(next_X)
                    front_dists = crowding_distance(F_combined, front)
                    order = np.argsort(-front_dists)
                    for i in range(min(remaining, len(front))):
                        idx = front[order[i]]
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                    break

            X = np.array(next_X[:pop_size])
            F = np.array(next_F[:pop_size])

        return X
