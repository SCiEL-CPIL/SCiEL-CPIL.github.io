"""Multi-objective optimizer for RE34 using NSGA-II style approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II with adaptive DE mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 315.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Initial population: structured + random
        pop_size = 50
        init_points = []

        # Corners
        for i in range(2 ** dim):
            point = np.array([hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)])
            init_points.append(point)

        # Midpoint
        init_points.append((lo + hi) / 2)

        # Face centers
        for j in range(dim):
            for k in range(j + 1, dim):
                for vj in [lo[j], hi[j]]:
                    for vk in [lo[k], hi[k]]:
                        point = np.array([(lo[t] + hi[t]) / 2 if t not in (j, k)
                                          else vj if t == j else vk for t in range(dim)])
                        init_points.append(point)

        # Fill with random
        while len(init_points) < pop_size:
            init_points.append(lo + (hi - lo) * rng.random(dim))

        X = np.array(init_points[:pop_size])
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Return True if a dominates b (a is better in all objectives, strictly in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_rank(F):
            """Return rank array and fronts (list of lists of indices)."""
            n = len(F)
            rank = np.zeros(n, dtype=int)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [set() for _ in range(n)]

            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        dominated_set[i].add(j)
                        domination_count[j] += 1
                    elif dominates(F[j], F[i]):
                        dominated_set[j].add(i)
                        domination_count[i] += 1

            fronts = []
            current_front = [i for i in range(n) if domination_count[i] == 0]
            current_rank = 0
            for i in current_front:
                rank[i] = current_rank

            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            rank[j] = current_rank + 1
                            next_front.append(j)
                current_front = next_front
                current_rank += 1

            return rank, fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            m = F[indices].shape[1]

            for k in range(m):
                obj_vals = F[indices, k]
                min_val = np.min(obj_vals)
                max_val = np.max(obj_vals)
                if max_val - min_val < 1e-12:
                    continue
                sorted_idx = np.argsort(obj_vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / (max_val - min_val)

            return cd

        # Main loop
        while used < budget:
            if time.time() - start_time > deadline:
                break

            # Adaptive offspring size
            n_offspring = min(15, budget - used)
            offspring_X = []
            offspring_F = []

            # Compute current rank for adaptive F factor
            rank, fronts = non_dominated_rank(F)

            for _ in range(n_offspring):
                if used >= budget:
                    break

                # Select 3 parents for DE mutation
                p1, p2, p3 = rng.integers(0, len(X), size=3)
                while p2 == p3:
                    p3 = rng.integers(len(X))

                # Adaptive F factor: larger for better-ranked parents
                base_F = 0.5
                F_factor = base_F + 0.3 * (1.0 - rank[p1] / max(1, len(fronts) - 1))
                F_factor = max(0.3, min(1.0, F_factor))
                CR = 0.7 + rng.random() * 0.3

                trial = X[p1].copy()
                r = rng.random(dim)
                for j in range(dim):
                    if r[j] < CR:
                        trial[j] = X[p1][j] + F_factor * (X[p2][j] - X[p3][j])
                    else:
                        trial[j] = X[p1][j]
                trial = np.clip(trial, lo, hi)

                ft = problem(trial)
                used += 1
                offspring_X.append(trial)
                offspring_F.append(ft)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Environmental selection
            n_keep = len(X)
            rank, fronts = non_dominated_rank(combined_F)

            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) <= n_keep:
                    selected_indices.extend(front)
                else:
                    remaining = n_keep - len(selected_indices)
                    cd = crowding_distance(combined_F, front)
                    sorted_by_cd = np.argsort(-cd)
                    for i in range(remaining):
                        selected_indices.append(front[sorted_by_cd[i]])
                    break

            X = combined_X[selected_indices]
            F = combined_F[selected_indices]

        return X
# EVOLVE-BLOCK-END
