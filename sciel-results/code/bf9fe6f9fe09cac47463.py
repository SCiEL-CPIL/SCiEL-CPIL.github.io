"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using a population-based approach with
    non-dominated sorting, crowding distance, and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(40, max(10, budget // 5))

        # Initialize population
        rng = np.random.default_rng(0)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume (given in task)
        ref = np.array([1.1, 1.3, 1.1])

        def fast_non_dominated_sort(F_arr):
            """Returns list of fronts, each a list of indices."""
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

            return fronts

        def crowding_distance(F_arr, indices):
            """Assign crowding distance to given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F_arr[indices, m]
                min_val = vals.min()
                max_val = vals.max()
                if max_val - min_val < 1e-15:
                    continue
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for k in range(1, n - 1):
                    dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (max_val - min_val)

            return dist

        # Main evolutionary loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Build next generation
            next_X = []
            next_F = []
            next_used = 0

            # Add all non-dominated points (first front)
            for i in fronts[0]:
                next_X.append(X[i])
                next_F.append(F[i])
                next_used += 1

            # Fill remaining slots with next fronts
            for front in fronts[1:]:
                if next_used >= pop_size:
                    break
                # Compute crowding distance for this front
                cd = crowding_distance(F, front)
                # Sort by crowding distance (descending)
                order = np.argsort(-cd)
                for idx in order:
                    if next_used >= pop_size:
                        break
                    i = front[idx]
                    next_X.append(X[i])
                    next_F.append(F[i])
                    next_used += 1

            # If we still need more, fill with random points
            while next_used < pop_size:
                x_new = lo + (hi - lo) * rng.random(dim)
                f_new = problem(x_new)
                next_X.append(x_new)
                next_F.append(f_new)
                next_used += 1
                used += 1

            X = np.array(next_X)
            F = np.array(next_F)

            # Generate offspring via mutation of non-dominated parents
            # Use adaptive mutation step size
            for _ in range(pop_size):
                if used >= budget or time.time() > deadline:
                    break

                # Select parent from first front (non-dominated)
                parent_idx = rng.integers(len(fronts[0]))
                parent = X[fronts[0][parent_idx]]

                # Adaptive mutation: start with larger steps, shrink over time
                progress = used / budget
                sigma = 0.3 * (1.0 - 0.5 * progress)
                child = parent + sigma * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                f_child = problem(child)
                used += 1

                # Replace worst point in population (by sum of objectives)
                worst_idx = np.argmax(np.sum(F, axis=1))
                X[worst_idx] = child
                F[worst_idx] = f_child

        return X
# EVOLVE-BLOCK-END
