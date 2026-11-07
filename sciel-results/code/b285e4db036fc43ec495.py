"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with deadline guard and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: keep moderate to allow good coverage within budget
        pop_size = min(40, max(20, budget // 5))

        # Initialize population
        X = lo + range_ * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_sort(F_arr):
            """Return list of fronts (list of indices)."""
            n = len(F_arr)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1

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

            remaining = [i for i in range(n) if i not in [j for front in fronts for j in front]]
            if remaining:
                fronts.append(remaining)

            return fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            for m in range(n_obj):
                values = F_arr[indices, m]
                mn, mx = values.min(), values.max()
                if mx - mn < 1e-12:
                    cd[:] = np.inf
                    continue
                order = np.argsort(values)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, len(order) - 1):
                    cd[order[k]] += (values[order[k + 1]] - values[order[k - 1]]) / (mx - mn)
            return cd

        def tournament_select(fronts, F, X, rng):
            """Select a parent using tournament on rank + crowding."""
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            crowd = np.zeros(n)
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(F, np.array(front))
                    for k, idx in enumerate(front):
                        crowd[idx] = cd[k]

            i1 = rng.integers(n)
            i2 = rng.integers(n)
            if ranks[i1] < ranks[i2]:
                return i1
            elif ranks[i2] < ranks[i1]:
                return i2
            else:
                return i1 if crowd[i1] > crowd[i2] else i2

        # Main evolution loop
        while used < budget:
            # Time check
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)

            # Select parent
            parent_idx = tournament_select(fronts, F, X, rng)
            parent = X[parent_idx]

            # Adaptive mutation: scale decreases over time for fine-tuning
            progress = used / budget
            mut_scale = 0.12 * (1.0 - 0.6 * progress)
            mutation = mut_scale * range_ * rng.standard_normal(dim)
            child = parent + mutation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst: highest rank, then lowest crowding
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            max_rank = ranks.max()
            worst_candidates = np.where(ranks == max_rank)[0]
            if len(worst_candidates) > 1:
                cd = crowding_distance(F, worst_candidates)
                worst_idx = worst_candidates[np.argmin(cd)]
            else:
                worst_idx = worst_candidates[0]

            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
