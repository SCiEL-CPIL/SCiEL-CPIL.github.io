import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        max_evals = self.budget

        # WFG9 is deceptive: the Pareto front is near the origin of the search space.
        # Initialize with a mix of uniform random and points near the origin.
        n_init = 40
        X = np.zeros((n_init, self.dim))
        # 20 uniform random
        X[:20] = np.random.uniform(lo, hi, (20, self.dim))
        # 10 very close to origin (within 10% of range)
        X[20:30] = lo + 0.1 * (hi - lo) * np.random.rand(10, self.dim)
        # 10 close to origin (within 5% of range)
        X[30:40] = lo + 0.05 * (hi - lo) * np.random.rand(10, self.dim)

        F = np.array([problem(x) for x in X])
        evals_used = n_init

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F_mat, n_total):
            ranks = np.zeros(n_total, dtype=int)
            S = [[] for _ in range(n_total)]
            n_dom = np.zeros(n_total, dtype=int)
            fronts = []

            for i in range(n_total):
                for j in range(n_total):
                    if i == j:
                        continue
                    if dominates(F_mat[i], F_mat[j]):
                        S[i].append(j)
                    elif dominates(F_mat[j], F_mat[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts.append([i])

            k = 0
            while k < len(fronts):
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break

            for r, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = r

            return ranks, fronts

        def crowding_distance(F_mat, indices):
            m = len(indices)
            if m <= 2:
                return np.full(m, np.inf)
            dist = np.zeros(m)
            F_sub = F_mat[indices]
            for k in range(self.n_obj):
                sorted_idx = np.argsort(F_sub[:, k])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], k] - F_sub[sorted_idx[0], k]
                if f_range > 1e-10:
                    for i in range(1, m - 1):
                        dist[sorted_idx[i]] += (F_sub[sorted_idx[i+1], k] - F_sub[sorted_idx[i-1], k]) / f_range
            return dist

        # Main loop
        while evals_used < max_evals:
            if time.time() > deadline:
                break

            # Generate offspring using DE/rand/1 with adaptive parameters
            pop_size = len(X)
            new_X = np.zeros((pop_size, self.dim))
            new_F = np.zeros((pop_size, self.n_obj))

            for i in range(pop_size):
                if evals_used >= max_evals:
                    break
                if time.time() > deadline:
                    break

                # Select 3 distinct parents
                while True:
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                    if r1 != r2 and r2 != r3 and r1 != r3 and r1 != i and r2 != i and r3 != i:
                        break

                # Adaptive F and CR
                F_factor = np.random.uniform(0.3, 0.9)
                CR = np.random.uniform(0.5, 0.95)

                v = X[r1] + F_factor * (X[r2] - X[r3])

                # Crossover
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        new_X[i, j] = v[j]
                    else:
                        new_X[i, j] = X[i, j]

                # Clip
                new_X[i] = np.clip(new_X[i], lo, hi)

                # Evaluate
                new_F[i] = problem(new_X[i])
                evals_used += 1

            # Combine and select
            all_X = np.vstack([X, new_X])
            all_F = np.vstack([F, new_F])

            n_total = len(all_F)
            ranks, fronts = non_dominated_sort(all_F, n_total)

            # Select best pop_size individuals
            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) <= pop_size:
                    selected_indices.extend(front)
                else:
                    remaining = pop_size - len(selected_indices)
                    dists = crowding_distance(all_F, front)
                    front_sorted = [front[i] for i in np.argsort(dists)[::-1]]
                    selected_indices.extend(front_sorted[:remaining])
                    break

            X = all_X[selected_indices]
            F = all_F[selected_indices]

            # Local search on a few best individuals to exploit WFG9's deceptive structure
            if evals_used < max_evals - 20:
                # Pick a few individuals with low objective sums
                sums = np.sum(F, axis=1)
                top_k = min(3, len(X))
                top_indices = np.argsort(sums)[:top_k]

                for idx in top_indices:
                    if evals_used >= max_evals - 5:
                        break
                    if time.time() > deadline:
                        break

                    best_x = X[idx].copy()
                    best_f = F[idx].copy()

                    # Coordinate descent with adaptive step
                    step_base = 0.05 * (hi - lo)
                    for _ in range(5):
                        if evals_used >= max_evals or time.time() > deadline:
                            break
                        d = np.random.randint(self.dim)
                        for sign in [1, -1]:
                            if evals_used >= max_evals:
                                break
                            trial_x = best_x.copy()
                            step = step_base[d] * np.random.uniform(0.1, 1.0)
                            trial_x[d] = np.clip(trial_x[d] + sign * step, lo[d], hi[d])
                            trial_f = problem(trial_x)
                            evals_used += 1
                            if dominates(trial_f, best_f):
                                best_x = trial_x
                                best_f = trial_f
                                break

                    X[idx] = best_x
                    F[idx] = best_f

        return X
