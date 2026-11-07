"""Multi-objective optimizer for DTLZ1 using a refined evolutionary strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective optimizer with adaptive mutation and archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(40, max(10, budget // 5))

        # Initialize population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_points = [F[i].copy() for i in range(pop_size)]

        # Archive for non-dominated points
        archive = list(range(pop_size))

        def dominates(a, b):
            """Check if a dominates b (all objectives <= and at least one <)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_indices(F_arr):
            """Return indices of non-dominated points in F_arr."""
            n = len(F_arr)
            nd = []
            for i in range(n):
                is_nd = True
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F_arr[j], F_arr[i]):
                        is_nd = False
                        break
                if is_nd:
                    nd.append(i)
            return nd

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for a set of indices."""
            if len(indices) <= 2:
                return np.ones(len(indices))
            n = len(indices)
            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F_arr[indices, m]
                sorted_idx = np.argsort(vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                val_range = vals[sorted_idx[-1]] - vals[sorted_idx[0]]
                if val_range > 0:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / val_range
            return cd

        # Main loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Identify non-dominated individuals in current population
            nd_idx = non_dominated_indices(F)

            # Select parents from non-dominated set (or all if none found)
            if len(nd_idx) == 0:
                nd_idx = list(range(pop_size))

            # Compute crowding distance for selection
            cd = crowding_distance(F, nd_idx)

            # Select two parents using tournament selection based on crowding
            parent1_idx = nd_idx[rng.integers(len(nd_idx))]
            parent2_idx = nd_idx[rng.integers(len(nd_idx))]

            # Crossover (SBX-like)
            alpha = 20
            child = np.zeros(dim)
            for d in range(dim):
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (alpha + 1))
                else:
                    beta = (2 * (1 - u)) ** (1 / (alpha + 1))
                child[d] = 0.5 * ((1 + beta) * X[parent1_idx, d] + (1 - beta) * X[parent2_idx, d])

            # Adaptive mutation
            # Estimate step size based on recent improvements
            step_size = 0.15 * (hi - lo)
            # Occasionally use larger jumps for exploration
            if rng.random() < 0.1:
                step_size = 0.4 * (hi - lo)
            elif rng.random() < 0.1:
                step_size = 0.05 * (hi - lo)

            child = child + step_size * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            # Evaluate child
            fc = problem(child)
            used += 1
            all_points.append(fc.copy())

            # Replace worst individual in population
            # Use a simple criterion: sum of objectives or distance to ideal point
            ideal = np.min(F, axis=0)
            distances = np.sum((F - ideal) ** 2, axis=1)
            worst_idx = np.argmax(distances)

            X[worst_idx] = child
            F[worst_idx] = fc

            # Update archive
            # Remove any archived points dominated by the new point
            new_archive = []
            for ai in archive:
                if not dominates(fc, F[ai]):
                    new_archive.append(ai)
            # Add new point if it's non-dominated w.r.t. archive
            is_dominated_by_archive = False
            for ai in new_archive:
                if dominates(F[ai], fc):
                    is_dominated_by_archive = True
                    break
            if not is_dominated_by_archive:
                new_archive.append(worst_idx)
            archive = new_archive

            # Limit archive size
            if len(archive) > 100:
                # Keep the most diverse points
                cd_archive = crowding_distance(F, archive)
                # Remove the point with smallest crowding distance
                min_cd_idx = np.argmin(cd_archive)
                archive.pop(min_cd_idx)

        return X
# EVOLVE-BLOCK-END
