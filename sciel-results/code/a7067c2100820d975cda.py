"""Multi-objective optimizer on WFG9 using a hybrid NSGA-II / differential-evolution approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid multi-objective optimizer: DE-style variation + NSGA-II selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Population size: 20 gives 10 full generations with budget=200
        pop_size = 20

        # Initialize population
        X = lo + span * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        def fast_non_dominated_sort(F_mat):
            """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
            n = len(F_mat)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_mat[i] <= F_mat[j]) and np.any(F_mat[i] < F_mat[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1

            current_front = [i for i in range(n) if domination_count[i] == 0]
            if current_front:
                fronts.append(current_front)

            while current_front:
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front

            return fronts

        def crowding_distance(F_mat, front_indices):
            """Compute crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            for m in range(n_obj):
                f_vals = F_mat[front_indices, m]
                min_val = f_vals.min()
                max_val = f_vals.max()
                if max_val - min_val < 1e-14:
                    dist[:] = np.inf
                    continue
                order = np.argsort(f_vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for k in range(1, n - 1):
                    dist[order[k]] += (f_vals[order[k + 1]] - f_vals[order[k - 1]]) / (max_val - min_val)
            return dist

        def select_next_generation(X_all, F_all, pop_size):
            """NSGA-II environmental selection."""
            fronts = fast_non_dominated_sort(F_all)
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    new_X.append(X_all[front])
                    new_F.append(F_all[front])
                else:
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(F_all, front)
                    order = np.argsort(-cd)
                    selected = [front[i] for i in order[:remaining]]
                    new_X.append(X_all[selected])
                    new_F.append(F_all[selected])
                    break
            if new_X:
                return np.vstack(new_X), np.vstack(new_F)
            return X_all[:pop_size], F_all[:pop_size]

        # Main evolutionary loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # DE-style offspring generation: for each individual, create a mutant
            # from 3 random distinct parents, then binomial crossover with the original
            offspring_X = []
            for i in range(pop_size):
                if time.time() > deadline:
                    break

                # Pick 3 distinct indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

                # DE/rand/1 mutation
                F_factor = 0.8
                mutant = X[r1] + F_factor * (X[r2] - X[r3])

                # Binomial crossover
                cross_rate = 0.5
                mask = np.random.random(dim) < cross_rate
                # Ensure at least one gene comes from mutant
                if not np.any(mask):
                    k = np.random.randint(dim)
                    mask[k] = True

                child = np.where(mask, mutant, X[i])
                child = np.clip(child, lo, hi)
                offspring_X.append(child)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array([problem(x) for x in offspring_X], dtype=np.float64)
            used += len(offspring_X)

            # Combine and select
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])
            X, F = select_next_generation(X_all, F_all, pop_size)

        return X
# EVOLVE-BLOCK-END
