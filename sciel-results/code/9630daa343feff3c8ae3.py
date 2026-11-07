"""Multi-objective optimizer for ZDT1 using a decomposition-based approach (NSGA-III style)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Decomposition-based multi-objective optimizer (NSGA-III inspired) for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Generate reference directions using Das-Dennis method for 2 objectives
        n_dirs = 199  # We'll use 199 directions to get good coverage
        directions = []
        for i in range(n_dirs + 1):
            directions.append([i / n_dirs, 1 - i / n_dirs])
        directions = np.array(directions)

        # Initial population: random points
        rng = np.random.default_rng(42)
        pop_size = min(199, budget // 2)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Assign each point to nearest direction (Tchebycheff decomposition)
        def assign_to_directions(F, directions):
            """Assign each point to its nearest reference direction using Tchebycheff distance."""
            n_pts = F.shape[0]
            n_dirs = len(directions)
            # Normalize objectives to [0, 1] range
            f_min = F.min(axis=0)
            f_max = F.max(axis=0)
            f_range = f_max - f_min
            f_range[f_range == 0] = 1
            F_norm = (F - f_min) / f_range

            # For each direction, find the point that minimizes Tchebycheff distance
            best_idx = np.zeros(n_dirs, dtype=int)
            best_dist = np.full(n_dirs, np.inf)

            for i, d in enumerate(directions):
                # Tchebycheff distance: max_j |d_j * (f_j - f*_j)|
                # Since we're working with normalized values and directions sum to 1,
                # we use: max_j |d_j * f_norm_j|
                dists = np.max(np.abs(directions[i] * F_norm), axis=1)
                best_idx[i] = np.argmin(dists)
                best_dist[i] = dists[best_idx[i]]

            return best_idx, best_dist

        # Evolutionary loop
        while used < budget:
            if time.time() > deadline - 10:
                break

            # Compute non-dominated set
            F_curr = F.copy()
            n = len(F_curr)

            # Simple non-dominated sorting
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(i + 1, n):
                    if dominated[j]:
                        continue
                    if np.all(F_curr[i] <= F_curr[j]) and np.any(F_curr[i] < F_curr[j]):
                        dominated[j] = True
                    elif np.all(F_curr[j] <= F_curr[i]) and np.any(F_curr[j] < F_curr[i]):
                        dominated[i] = True
                        break

            non_dom_mask = ~dominated
            n_non_dom = np.sum(non_dom_mask)

            # Select parents from non-dominated set
            if n_non_dom > 0:
                parent_indices = np.where(non_dom_mask)[0]
            else:
                parent_indices = np.arange(n)

            # Generate multiple children per iteration for efficiency
            n_children = min(10, budget - used)
            if n_children <= 0:
                break

            for _ in range(n_children):
                if time.time() > deadline - 5:
                    break

                # Select parent
                parent_idx = parent_indices[rng.integers(len(parent_indices))]
                parent = X[parent_idx].copy()

                # Mutation: Gaussian perturbation
                sigma = 0.1 * (hi - lo)
                child = parent + sigma * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                # Evaluate
                fc = problem(child)
                used += 1

                # Replace worst point based on sum of objectives
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = child
                F[worst] = fc

        return X
# EVOLVE-BLOCK-END
