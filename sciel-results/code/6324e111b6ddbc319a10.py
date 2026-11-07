"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        # Population size
        pop_size = 40
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        def is_dominated(a, b):
            """Return True if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """Return fronts as list of lists of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # set of indices dominated by i
            nd = [0] * n  # number of solutions that dominate i
            rank = [0] * n
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if is_dominated(F[i], F[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif is_dominated(F[j], F[i]):
                        S[j].append(i)
                        nd[i] += 1
                if nd[i] == 0:
                    rank[i] = 0
                    fronts.append([i])

            k = 0
            while k < len(fronts):
                current_front = fronts[k]
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for indices in a front."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            dist = [0.0] * n
            for m in range(self.n_obj):
                sorted_idx = sorted(range(n), key=lambda i: F[front_indices[i], m])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                f_min = F[front_indices[sorted_idx[0]], m]
                f_max = F[front_indices[sorted_idx[-1]], m]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (
                        F[front_indices[sorted_idx[i + 1]], m] -
                        F[front_indices[sorted_idx[i - 1]], m]
                    ) / f_range
            return dist

        def select_parents(fronts, F, n_parents):
            """Select n_parents from fronts using crowding distance."""
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= n_parents:
                    selected.extend(front)
                else:
                    remaining = n_parents - len(selected)
                    cd = crowding_distance(F, front)
                    sorted_by_cd = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    for i in sorted_by_cd[:remaining]:
                        selected.append(front[i])
                    break
            return selected

        # Main loop
        while used < self.budget:
            if time.time() > deadline - 5:
                break

            fronts = fast_non_dominated_sort(F)

            # DE parameters
            CR = 0.9
            F_DE = 0.8

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, self.budget - used)

            for _ in range(n_offspring):
                # Select 3 distinct parents from the whole population
                idx = rng.choice(pop_size, size=3, replace=False)
                a, b, c = idx[0], idx[1], idx[2]
                v = X[a] + F_DE * (X[b] - X[c])
                v = np.clip(v, lo, hi)
                fv = problem(v)
                offspring_X.append(v)
                offspring_F.append(fv)
                used += 1
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break

            if not offspring_X:
                break

            offspring_F = np.array(offspring_F, dtype=np.float64)
            offspring_X = np.array(offspring_X, dtype=np.float64)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_F)

            # Non-dominated sort on combined
            fronts = fast_non_dominated_sort(combined_F)

            # Select next population
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for i in front:
                        next_X.append(combined_X[i])
                        next_F.append(combined_F[i])
                else:
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(combined_F, front)
                    sorted_by_cd = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    for i in sorted_by_cd[:remaining]:
                        next_X.append(combined_X[front[i]])
                        next_F.append(combined_F[front[i]])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
