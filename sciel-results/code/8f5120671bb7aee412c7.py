# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using a population-based evolutionary strategy
    with non-dominated sorting, crowding distance, and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        # Population size: use a reasonable chunk of budget
        # With 200 evals, pop of ~20-40 is reasonable
        pop_size = min(30, max(10, self.budget // 7))
        # But we need at least 2 for crossover
        pop_size = max(2, pop_size)

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track best known non-dominated set for return
        # We'll maintain an archive of all non-dominated points seen

        def fast_non_dominated_sort(F):
            """Return list of fronts, each front is list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dominating = np.zeros(n, dtype=int)
            fronts = []
            front0 = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominating[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
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

        while used < self.budget:
            if time.time() > deadline:
                break

            # Compute fronts
            fronts = fast_non_dominated_sort(F)

            # Select parents from best fronts
            parents = []
            for front in fronts:
                parents.extend(front)
                if len(parents) >= pop_size // 2:
                    break
            if len(parents) < 2:
                parents = list(range(len(F)))
            parents = parents[:max(2, pop_size // 2)]

            # Generate children via mutation of random parents
            n_children = min(pop_size, self.budget - used)
            if n_children <= 0:
                break

            X_c = np.zeros((n_children, self.dim))
            for i in range(n_children):
                p_idx = parents[rng.integers(len(parents))]
                # Adaptive mutation step: start larger, shrink over time
                progress = used / self.budget
                step = (1.0 - 0.5 * progress) * 0.15 * (hi - lo)
                X_c[i] = X[p_idx] + step * rng.standard_normal(self.dim)
                X_c[i] = np.clip(X_c[i], lo, hi)

            F_c = np.array([problem(x) for x in X_c])
            used += n_children

            # Combine and select next population
            X_all = np.vstack([X, X_c])
            F_all = np.vstack([F, F_c])
            fronts = fast_non_dominated_sort(F_all)

            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    new_X.append(front)
                    new_F.append(front)
                else:
                    # Need to truncate this front
                    remaining = pop_size - len([idx for f in new_X for idx in f])
                    if remaining <= 0:
                        break
                    cd = crowding_distance(F_all, front)
                    sorted_front = [front[j] for j in np.argsort(-cd)]
                    new_X.append(sorted_front[:remaining])
                    new_F.append(sorted_front[:remaining])
                    break

            # Flatten
            all_indices = [idx for f in new_X for idx in f]
            X = X_all[all_indices]
            F = F_all[all_indices]

        return X
# EVOLVE-BLOCK-END
