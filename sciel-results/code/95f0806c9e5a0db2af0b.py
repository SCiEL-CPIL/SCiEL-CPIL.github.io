# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II with adaptive mutation and
    crowding distance selection, plus a deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        # Population size
        pop_size = 24
        if pop_size > self.budget:
            pop_size = max(2, self.budget)

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Return True if a dominates b (minimization)."""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """Return list of fronts, each front is list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominating = np.zeros(n, dtype=int)
            fronts = []
            front0 = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_dominating[j] += 1
                    elif dominates(F[j], F[i]):
                        n_dominating[i] += 1
                if n_dominating[i] == 0:
                    front0.append(i)

            fronts.append(front0)
            while front0:
                next_front = []
                for i in front0:
                    for j in S[i]:
                        n_dominating[j] -= 1
                        if n_dominating[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front0 = next_front
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for points in front_indices."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(self.n_obj):
                vals = F[front_indices, m]
                min_v, max_v = vals.min(), vals.max()
                if max_v - min_v < 1e-15:
                    continue
                sorted_idx = np.argsort(vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / (max_v - min_v)
            return cd

        def select_parents(fronts, F, n_parents):
            """Select n_parents indices from fronts, preferring lower fronts."""
            parents = []
            for front in fronts:
                if len(parents) + len(front) <= n_parents:
                    parents.extend(front)
                else:
                    remaining = n_parents - len(parents)
                    if remaining <= 0:
                        break
                    cd = crowding_distance(F, front)
                    sorted_front = [front[j] for j in np.argsort(-cd)]
                    parents.extend(sorted_front[:remaining])
                    break
            return parents[:n_parents]

        while used < self.budget:
            if time.time() > deadline:
                break

            # Compute fronts for current population
            fronts = fast_non_dominated_sort(F)

            # Number of children to generate
            n_children = min(pop_size, self.budget - used)
            if n_children <= 0:
                break

            # Select parents
            n_parents = max(2, n_children)
            parents = select_parents(fronts, F, n_parents)

            # Generate children via SBX-like crossover + polynomial mutation
            X_c = np.zeros((n_children, self.dim))
            for i in range(n_children):
                # Pick two parents
                p1_idx = parents[rng.integers(len(parents))]
                p2_idx = parents[rng.integers(len(parents))]
                p1 = X[p1_idx]
                p2 = X[p2_idx]

                # SBX crossover
                eta_c = 20.0
                child = np.zeros(self.dim)
                for d in range(self.dim):
                    if rng.random() < 0.5:
                        if abs(p1[d] - p2[d]) < 1e-14:
                            child[d] = p1[d]
                        else:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1.0 / (eta_c + 1))
                            else:
                                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                            child[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
                    else:
                        if abs(p1[d] - p2[d]) < 1e-14:
                            child[d] = p1[d]
                        else:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1.0 / (eta_c + 1))
                            else:
                                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                            child[d] = 0.5 * ((1 + beta) * p2[d] + (1 - beta) * p1[d])

                # Polynomial mutation
                eta_m = 20.0
                progress = used / self.budget
                # Adaptive mutation rate: higher early, lower later
                mut_rate = 1.0 / self.dim * (1.5 - 0.5 * progress)
                for d in range(self.dim):
                    if rng.random() < mut_rate:
                        u = rng.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                        child[d] += delta * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                X_c[i] = child

            F_c = np.array([problem(x) for x in X_c])
            used += n_children

            # Combine and select next population
            X_all = np.vstack([X, X_c])
            F_all = np.vstack([F, F_c])
            fronts = fast_non_dominated_sort(F_all)

            # Select pop_size individuals from combined population
            new_indices = []
            for front in fronts:
                if len(new_indices) + len(front) <= pop_size:
                    new_indices.extend(front)
                else:
                    remaining = pop_size - len(new_indices)
                    if remaining <= 0:
                        break
                    cd = crowding_distance(F_all, front)
                    sorted_front = [front[j] for j in np.argsort(-cd)]
                    new_indices.extend(sorted_front[:remaining])
                    break

            # If we still need more (shouldn't happen but safety)
            while len(new_indices) < pop_size:
                new_indices.append(rng.integers(len(F_all)))

            X = X_all[new_indices]
            F = F_all[new_indices]

        return X
# EVOLVE-BLOCK-END
