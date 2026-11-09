"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using a simple NSGA-II-like approach for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(50, max(10, self.budget // 4))

        # Initial population
        X = np.empty((pop_size, self.dim))
        for i in range(pop_size):
            X[i] = lo + (hi - lo) * rng.random(self.dim)

        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_sort(F_arr):
            """Perform non-dominated sorting and return list of fronts (each a list of indices)."""
            n = len(F_arr)
            if n == 0:
                return []
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = []
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F_arr[p] <= F_arr[q]) and np.any(F_arr[p] < F_arr[q]):
                        S[p].append(q)
                    elif np.all(F_arr[q] <= F_arr[p]) and np.any(F_arr[q] < F_arr[p]):
                        nd[p] += 1
                if nd[p] == 0:
                    fronts.append([p])
            i = 0
            while i < len(fronts) and fronts[i]:
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

        def crowding_distance(F_arr, front_indices):
            """Compute crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(self.n_obj):
                sorted_idx = np.argsort(F_arr[front_indices, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_min = F_arr[front_indices[sorted_idx[0]], m]
                f_max = F_arr[front_indices[sorted_idx[-1]], m]
                if f_max - f_min > 1e-12:
                    for k in range(1, n - 1):
                        cd[sorted_idx[k]] += (F_arr[front_indices[sorted_idx[k + 1]], m] -
                                               F_arr[front_indices[sorted_idx[k - 1]], m]) / (f_max - f_min)
            return cd

        while used < self.budget:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring_X = []
            offspring_F = []
            for i in range(pop_size):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Tournament selection
                fronts = non_dominated_sort(F)
                # Pick a front, favoring earlier ones
                total = sum(len(f) for f in fronts)
                r = rng.random() * total
                cum = 0
                selected_front = fronts[0]
                for fi in fronts:
                    cum += len(fi)
                    if r <= cum:
                        selected_front = fi
                        break
                parent_idx = selected_front[rng.integers(len(selected_front))]
                parent = X[parent_idx]

                # Mutation
                child = parent.copy()
                for j in range(self.dim):
                    if rng.random() < 0.9:
                        delta = rng.random()
                        if delta < 0.5:
                            alpha = (2 * delta) ** 0.05
                            sign = -1
                        else:
                            alpha = (2 * (1 - delta)) ** 0.05
                            sign = 1
                        child[j] = parent[j] + sign * alpha * (hi[j] - lo[j])
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parents and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])

            # Non-dominated sort on combined population
            fronts = non_dominated_sort(F_all)

            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    new_X.extend(X_all[front])
                    new_F.extend(F_all[front])
                else:
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(F_all, front)
                    sorted_by_cd = np.argsort(-cd)
                    selected = [front[i] for i in sorted_by_cd[:remaining]]
                    new_X.extend(X_all[selected])
                    new_F.extend(F_all[selected])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X

# EVOLVE-BLOCK-END
