"""Multi-objective optimizer for DTLZ4 using NSGA-II with time-aware budget management."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 40s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Population size
        pop_size = min(100, max(20, self.budget // 2))

        # Random initialization
        rng = np.random.default_rng(0)
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # DTLZ4 specific mutation: bias towards low values
        # DTLZ4 has a non-linear distribution along the Pareto front
        # The transformation x_i = u_i^(1/beta) where beta=100 for DTLZ4
        # So we should bias our search towards lower values

        def mutate(parent):
            # Crossover-like: mix with random point
            child = parent.copy()
            # Apply DTLZ4 bias: transform to emphasize low values
            # Use a power transformation to bias towards 0
            mask = rng.random(self.dim) < 0.5
            if np.any(mask):
                child[mask] = lo[mask] + range_[mask] * rng.random(np.sum(mask)) ** 100
            # Add some noise
            noise_scale = 0.05 * range_
            child += rng.standard_normal(self.dim) * noise_scale
            return np.clip(child, lo, hi)

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of solutions dominating i
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    # Check if j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1

            # First front
            front = [i for i in range(n) if n_dom[i] == 0]
            fronts.append(front)

            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                front = next_front
                if front:
                    fronts.append(front)

            return fronts

        def crowding_distance(F, front_indices):
            """Calculate crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            n_obj = F.shape[1]

            for m in range(n_obj):
                vals = F[front_indices, m]
                min_val, max_val = vals.min(), vals.max()
                if max_val - min_val < 1e-12:
                    cd[:] = np.inf
                    continue
                # Sort by objective m
                sorted_idx = np.argsort(vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / (max_val - min_val)

            return cd

        # Main evolution loop
        while used < self.budget:
            # Check time
            if time.time() > deadline - 5:
                break

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Select parents from first front
            if len(fronts) > 0 and len(fronts[0]) > 0:
                parent_idx = fronts[0][rng.integers(len(fronts[0]))]
            else:
                parent_idx = rng.integers(pop_size)

            # Create child
            child = mutate(X[parent_idx])
            fc = problem(child)
            used += 1

            # Replace worst point (by sum of objectives)
            worst_idx = np.argmax(np.sum(F, axis=1))
            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
