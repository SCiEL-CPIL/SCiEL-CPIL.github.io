"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 combining NSGA-II selection with
    differential evolution operators and hypervolume-aware mutation."""

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
        pop_size = 30

        # Initialize population
        rng = np.random.default_rng(0)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume (given in task)
        ref = np.array([1.1, 1.3, 1.1])

        def dominates(a, b):
            """Check if vector a dominates vector b (minimization)."""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F_arr):
            """Returns list of fronts, each a list of indices."""
            n = len(F_arr)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F_arr[i], F_arr[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif dominates(F_arr[j], F_arr[i]):
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

        def hypervolume_contribution(point, points, ref_point):
            """
            Approximate the hypervolume contribution of a single point
            relative to a set of other points and a reference point.
            Returns 0 if point is dominated by any point in the set.
            """
            # If any point dominates this point, contribution is 0
            for p in points:
                if dominates(p, point):
                    return 0.0

            # Calculate the volume of the box where this point is the best
            # in each dimension, bounded by the reference point and
            # the best values seen in other points for each dimension
            vol = 1.0
            for d in range(n_obj):
                # Find the minimum value in dimension d among all other points
                # that don't dominate this point (or all points if none dominate)
                min_other = ref_point[d]
                for p in points:
                    if p[d] < point[d]:
                        min_other = min(min_other, p[d])
                # The contribution in this dimension is from point[d] to min_other
                # but capped by ref_point[d]
                extent = min_other - point[d]
                if extent < 0:
                    extent = 0.0
                vol *= extent

            return vol

        # Main evolutionary loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Build next generation using NSGA-II style selection
            next_indices = []
            next_used = 0

            # Add all non-dominated points (first front)
            for i in fronts[0]:
                next_indices.append(i)
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
                    next_indices.append(i)
                    next_used += 1

            # If we still need more, fill with random points
            while next_used < pop_size:
                x_new = lo + (hi - lo) * rng.random(dim)
                f_new = problem(x_new)
                next_indices.append(-1)  # Mark as new random
                next_used += 1
                used += 1

            # Build new population
            new_X = []
            new_F = []
            for i in next_indices:
                if i == -1:
                    x_new = lo + (hi - lo) * rng.random(dim)
                    f_new = problem(x_new)
                    new_X.append(x_new)
                    new_F.append(f_new)
                else:
                    new_X.append(X[i])
                    new_F.append(F[i])

            X = np.array(new_X)
            F = np.array(new_F)

            # Generate offspring via differential evolution with adaptive parameters
            progress = used / budget
            cr = 0.9
            # Adaptive scale factor: start higher, decrease over time
            scale = 0.8 * (1.0 - 0.5 * progress)

            for _ in range(pop_size):
                if used >= budget or time.time() > deadline:
                    break

                # Select three distinct random parents
                idxs = rng.choice(pop_size, size=3, replace=False)
                base = X[idxs[0]]
                v = base + scale * (X[idxs[1]] - X[idxs[2]])
                v = np.clip(v, lo, hi)

                # Crossover
                trial = X[rng.integers(pop_size)].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < cr or j == j_rand:
                        trial[j] = v[j]

                f_trial = problem(trial)
                used += 1

                # Selection: keep the better of parent and trial
                parent_idx = rng.integers(pop_size)
                if dominates(f_trial, F[parent_idx]):
                    X[parent_idx] = trial
                    F[parent_idx] = f_trial
                elif not dominates(F[parent_idx], f_trial):
                    # Non-dominated: use crowding distance heuristic
                    # Keep trial if it's in a less crowded region
                    if rng.random() < 0.5:
                        X[parent_idx] = trial
                        F[parent_idx] = f_trial

        return X
# EVOLVE-BLOCK-END
