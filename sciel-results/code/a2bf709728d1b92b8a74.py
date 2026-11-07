"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        N = 24
        if N > self.budget:
            N = max(2, self.budget)

        # Initial population
        rng = np.random.default_rng(0)
        X = lo + span * rng.random((N, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = N

        def dominates(a, b):
            """True if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Returns list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_count[i] += 1
                if n_count[i] == 0:
                    fronts.append([i])

            k = 0
            while k < len(fronts):
                current = fronts[k]
                next_front = []
                for i in current:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F[indices, m]
                min_v, max_v = vals.min(), vals.max()
                if max_v - min_v < 1e-15:
                    continue
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for i in range(1, n - 1):
                    dist[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (max_v - min_v)
            return dist

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)

            # Create offspring
            offspring_X = []
            offspring_F = []

            # Selection via tournament on ranks
            ranks = np.zeros(N, dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            # Generate N offspring
            for _ in range(N):
                # Tournament selection (size 2)
                i1 = rng.integers(N)
                i2 = rng.integers(N)
                if ranks[i1] < ranks[i2]:
                    parent1 = i1
                elif ranks[i2] < ranks[i1]:
                    parent1 = i2
                else:
                    parent1 = i1 if rng.random() < 0.5 else i2

                i1 = rng.integers(N)
                i2 = rng.integers(N)
                if ranks[i1] < ranks[i2]:
                    parent2 = i1
                elif ranks[i2] < ranks[i1]:
                    parent2 = i2
                else:
                    parent2 = i1 if rng.random() < 0.5 else i2

                # SBX crossover
                if rng.random() < 0.9:
                    child1 = np.empty(dim)
                    child2 = np.empty(dim)
                    for d in range(dim):
                        if rng.random() < 0.5:
                            # SBX
                            u = rng.random()
                            eta = 20.0
                            if u <= 0.5:
                                beta = (2 * u) ** (1.0 / (eta + 1))
                            else:
                                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                            child1[d] = 0.5 * ((1 + beta) * X[parent1, d] + (1 - beta) * X[parent2, d])
                            child2[d] = 0.5 * ((1 - beta) * X[parent1, d] + (1 + beta) * X[parent2, d])
                        else:
                            child1[d] = X[parent1, d]
                            child2[d] = X[parent2, d]
                    # Polynomial mutation
                    for d in range(dim):
                        if rng.random() < 1.0 / dim:
                            u = rng.random()
                            eta_m = 20.0
                            if u <= 0.5:
                                delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                            else:
                                delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                            child1[d] += delta * (hi[d] - lo[d])
                        if rng.random() < 1.0 / dim:
                            u = rng.random()
                            eta_m = 20.0
                            if u <= 0.5:
                                delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                            else:
                                delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                            child2[d] += delta * (hi[d] - lo[d])
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                    offspring_X.append(child1)
                    offspring_X.append(child2)
                    offspring_F.append(problem(child1))
                    offspring_F.append(problem(child2))
                    used += 2
                else:
                    # Just copy with mutation
                    child = X[parent1].copy()
                    for d in range(dim):
                        if rng.random() < 1.0 / dim:
                            u = rng.random()
                            eta_m = 20.0
                            if u <= 0.5:
                                delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                            else:
                                delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                            child[d] += delta * (hi[d] - lo[d])
                    child = np.clip(child, lo, hi)
                    offspring_X.append(child)
                    offspring_F.append(problem(child))
                    used += 1

                if used >= self.budget:
                    break

            if not offspring_X:
                break

            offspring_F = np.array(offspring_F, dtype=np.float64)
            offspring_X = np.array(offspring_X, dtype=np.float64)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            combined_N = len(combined_F)

            # Non-dominated sort on combined
            fronts = non_dominated_sort(combined_F)

            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= N:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to use crowding distance to pick
                    remaining = N - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    order = np.argsort(-cd)
                    for pos in order[:remaining]:
                        idx = front[pos]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X, dtype=np.float64)
            F = np.array(new_F, dtype=np.float64)

        return X
# EVOLVE-BLOCK-END
