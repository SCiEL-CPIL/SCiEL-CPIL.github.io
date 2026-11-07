import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        n = self.dim
        m = self.n_obj
        budget = self.budget

        N = 80

        # Initialize population
        X = lo + span * rng.random((N, n))
        F = np.array([problem(x) for x in X], dtype=float)
        used = N

        # Reference point (ideal point approximation)
        ideal = F.min(axis=0)

        # Crowding distance function
        def crowding_distance(F_sub, indices):
            d = np.zeros(len(indices), dtype=float)
            if len(indices) <= 2:
                return d
            for k in range(m):
                f_vals = F_sub[indices, k]
                f_min = f_vals.min()
                f_max = f_vals.max()
                if f_max - f_min < 1e-14:
                    continue
                order = np.argsort(f_vals)
                d[order[0]] = np.inf
                d[order[-1]] = np.inf
                for j in range(1, len(order) - 1):
                    d[order[j]] += (f_vals[order[j + 1]] - f_vals[order[j - 1]]) / (f_max - f_min)
            return d

        # Non-dominated sorting
        def non_dominated_sort(F_arr):
            n_pts = len(F_arr)
            rank = np.zeros(n_pts, dtype=int)
            fronts = []
            dominated_by = np.zeros(n_pts, dtype=int)
            dominates = [[] for _ in range(n_pts)]

            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominates[i].append(j)
                        dominated_by[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominates[j].append(i)
                        dominated_by[i] += 1

            current_front = [i for i in range(n_pts) if dominated_by[i] == 0]
            rank_idx = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominates[i]:
                        dominated_by[j] -= 1
                        if dominated_by[j] == 0:
                            rank[j] = rank_idx
                            next_front.append(j)
                rank_idx += 1
                current_front = next_front
            return fronts

        # BLX-Alpha crossover
        def blx_alpha(x1, x2, alpha=0.5):
            a = np.minimum(x1, x2)
            b = np.maximum(x1, x2)
            child = a + rng.uniform(-alpha, 1 + alpha, n) * (b - a)
            return np.clip(child, lo, hi)

        # Gaussian mutation
        def mutate(x, sigma=0.1):
            return np.clip(x + rng.normal(0, sigma, n), lo, hi)

        # Evolution loop
        while used < budget:
            # Non-dominated sorting
            fronts = non_dominated_sort(F)
            dist = np.zeros(N, dtype=float)
            for front in fronts:
                dist[front] = crowding_distance(F, front)

            # Create mating pool using tournament selection
            def tournament_select():
                a = rng.integers(0, N)
                b = rng.integers(0, N)
                # Compare by rank then crowding distance
                if rank[a] < rank[b]:
                    return a
                elif rank[b] < rank[a]:
                    return b
                else:
                    if dist[a] > dist[b]:
                        return a
                    else:
                        return b

            # Generate offspring
            while used < budget:
                p1 = tournament_select()
                p2 = tournament_select()
                if p1 == p2:
                    p2 = (p2 + 1) % N

                child = blx_alpha(X[p1], X[p2])
                child = mutate(child)

                f_child = problem(child)
                used += 1

                # Add to population (elitism: keep best N)
                X = np.vstack([X, child])
                F = np.vstack([F, f_child])

                # Truncate to N best
                fronts_new = non_dominated_sort(F)
                dist_new = np.zeros(len(F), dtype=float)
                for front in fronts_new:
                    dist_new[front] = crowding_distance(F, front)

                # Select top N
                selected = []
                for front in fronts_new:
                    if len(selected) + len(front) <= N:
                        selected.extend(front)
                    else:
                        front_dist = dist_new[front]
                        order = np.argsort(front_dist)[::-1]
                        for idx in order:
                            if len(selected) < N:
                                selected.append(front[idx])
                            else:
                                break
                        break

                X = X[selected]
                F = F[selected]
                # Recalculate rank and dist for next iteration
                fronts = non_dominated_sort(F)
                dist = np.zeros(N, dtype=float)
                for front in fronts:
                    dist[front] = crowding_distance(F, front)
                rank = np.zeros(N, dtype=int)
                for i, front in enumerate(fronts):
                    rank[front] = i

        return X
