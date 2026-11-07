"""Multi-objective optimizer for WFG4 with deadline guard."""
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
        deadline = start_time + 280.0  # 280s safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # Population size: use a good fraction of budget
        pop_size = min(50, max(20, self.budget // 5))

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        def check_deadline():
            return time.time() >= deadline

        def non_dominated_sort(F_arr):
            """Return list of fronts (indices) from non-dominated sorting."""
            n = len(F_arr)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1

            current_front = [i for i in range(n) if domination_count[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front

            # If some points remain (shouldn't happen but safety)
            remaining = [i for i in range(n) if i not in set(idx for f in fronts for idx in f)]
            if remaining:
                fronts.append(remaining)

            return fronts

        def crowding_distance(F_arr, front_indices):
            """Compute crowding distance for indices in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n, dtype=np.float64)
            F_front = F_arr[front_indices]

            for m in range(self.n_obj):
                sorted_idx = np.argsort(F_front[:, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = F_front[sorted_idx[-1], m] - F_front[sorted_idx[0], m]
                if f_range > 0:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (F_front[sorted_idx[i + 1], m] - F_front[sorted_idx[i - 1], m]) / f_range

            return cd

        # Main loop
        while used < self.budget:
            if check_deadline():
                break

            # Non-dominated sorting
            fronts = non_dominated_sort(F)

            # Build mating pool from first fronts
            mating_pool = []
            for front in fronts:
                if len(mating_pool) + len(front) >= pop_size:
                    # Need crowding distance for this front
                    cd = crowding_distance(F, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    mating_pool.extend(sorted_front[:pop_size - len(mating_pool)])
                    break
                else:
                    mating_pool.extend(front)

            if len(mating_pool) == 0:
                mating_pool = list(range(pop_size))

            # Generate offspring
            n_offspring = min(pop_size, self.budget - used)
            offspring_X = []
            offspring_F = []

            for _ in range(n_offspring):
                if check_deadline():
                    break

                # Binary tournament
                p1 = mating_pool[rng.integers(len(mating_pool))]
                p2 = mating_pool[rng.integers(len(mating_pool))]
                # Select based on which is in earlier front (simplified: just pick one)
                parent = X[p1] if rng.random() < 0.5 else X[p2]

                # Mutation: Gaussian with adaptive step
                step = 0.1 * span
                child = parent + step * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)

                # SBX-style crossover occasionally
                if rng.random() < 0.5 and len(mating_pool) >= 2:
                    p1 = mating_pool[rng.integers(len(mating_pool))]
                    p2 = mating_pool[rng.integers(len(mating_pool))]
                    if p1 != p2:
                        alpha = rng.random(self.dim)
                        child = alpha * X[p1] + (1 - alpha) * X[p2]
                        child = child + 0.05 * span * rng.standard_normal(self.dim)
                        child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

            if len(offspring_X) == 0:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_F)

            # Non-dominated sort on combined
            fronts = non_dominated_sort(combined_F)

            # Select next generation
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    next_X.extend(combined_X[front])
                    next_F.extend(combined_F[front])
                else:
                    # Need to trim this front by crowding distance
                    cd = crowding_distance(combined_F, front)
                    needed = pop_size - len(next_X)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    next_X.extend(combined_X[sorted_front[:needed]])
                    next_F.extend(combined_F[sorted_front[:needed]])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
