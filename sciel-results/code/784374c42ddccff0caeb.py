# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-exploiting optimizer for DTLZ4 with adaptive mutation and polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.n_shape = n_obj - 1
        self.n_dist = dim - self.n_shape

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ4 structure: first M-1 vars are shape, rest are distance
        x_dist_opt = 0.5 * (lo[self.n_shape:] + hi[self.n_shape:])

        pop_size = 60
        used = 0

        # Initialize with quasi-random spread over the shape simplex
        # For 3 objectives, shape vars (x0, x1) with x0+x1 <= 1 map to the front
        # Use Sobol-like or stratified random for good coverage
        X = np.zeros((pop_size, self.dim))
        # Stratified random over [0,1]^2 for shape vars
        n_per_cell = max(1, pop_size // 4)
        cells = []
        for i in range(2):
            for j in range(2):
                cells.append((i / 2.0, j / 2.0))
        # Fill remaining with random
        while len(cells) < pop_size:
            cells.append((np.random.rand(), np.random.rand()))
        
        for k in range(pop_size):
            c = cells[k]
            X[k, 0] = lo[0] + (hi[0] - lo[0]) * c[0]
            X[k, 1] = lo[1] + (hi[1] - lo[1]) * c[1]
            X[k, self.n_shape:] = x_dist_opt
        
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Vectorized dominance helpers
        def is_dominated_by_pop(f, F_pop):
            leq = np.all(F_pop <= f, axis=1)
            lt = np.any(F_pop < f, axis=1)
            return np.any(leq & lt)

        def dominated_indices_in_pop(f, F_pop):
            leq = np.all(f <= F_pop, axis=1)
            lt = np.any(f < F_pop, axis=1)
            return np.where(leq & lt)[0]

        def compute_crowding(F_pop, indices):
            """Compute crowding distance for given indices in F_pop."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            F_sub = F_pop[indices]
            for m in range(self.n_obj):
            # Sort by objective m
                order = np.argsort(F_sub[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for i in range(1, n - 1):
                    if F_sub[order[i+1], m] - F_sub[order[i-1], m] > 1e-15:
                        cd[order[i]] += (F_sub[order[i+1], m] - F_sub[order[i-1], m])
            return cd

        # Main search loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Adaptive sigma: reduce as we approach the end
            progress = used / self.budget
            # Start with larger sigma, reduce to very small for polish
            sigma_shape = 0.15 * (1.0 - 0.7 * progress)
            sigma_dist = 0.06 * (1.0 - 0.6 * progress)

            # Select parent: prefer non-dominated, use crowding for diversity
            # Find non-dominated set
            nd_mask = np.array([not is_dominated_by_pop(F[i], F) for i in range(pop_size)])
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) > 0:
                # Use crowding distance to pick diverse parent
                cd = compute_crowding(F, nd_indices)
                # Pick parent with highest crowding distance (most diverse)
                parent_local_idx = nd_indices[np.argmax(cd)]
            else:
                parent_local_idx = np.random.randint(pop_size)
            
            parent = X[parent_local_idx]

            # Create child
            child = parent.copy()

            # Mutate shape variables
            child[:self.n_shape] += sigma_shape * (hi[:self.n_shape] - lo[:self.n_shape]) * np.random.randn(self.n_shape)
            child[:self.n_shape] = np.clip(child[:self.n_shape], lo[:self.n_shape], hi[:self.n_shape])

            # Mutate distance variables: reset near optimum with small noise
            child[self.n_shape:] = x_dist_opt + sigma_dist * (hi[self.n_shape:] - lo[self.n_shape:]) * np.random.randn(self.n_dist)
            child[self.n_shape:] = np.clip(child[self.n_shape:], lo[self.n_shape:], hi[self.n_shape:])

            f_child = problem(child)
            used += 1

            # Update population
            if not is_dominated_by_pop(f_child, F):
                dom_idx = dominated_indices_in_pop(f_child, F)
                if len(dom_idx) > 0:
                    # Replace a dominated point, prefer the one with worst hypervolume contribution
                    # Approximate: highest sum of objectives
                    sums = np.sum(F[dom_idx], axis=1)
                    rep_idx = dom_idx[np.argmax(sums)]
                    X[rep_idx] = child
                    F[rep_idx] = f_child
                else:
                    # Not dominated, doesn't dominate: replace the point with least crowding
                    # among non-dominated, or worst sum overall
                    nd_mask2 = np.array([not is_dominated_by_pop(F[i], F) for i in range(pop_size)])
                    nd_idx2 = np.where(nd_mask2)[0]
                    if len(nd_idx2) > 0:
                        cd2 = compute_crowding(F, nd_idx2)
                        # Replace the one with smallest crowding distance
                        rep_idx = nd_idx2[np.argmin(cd2)]
                        X[rep_idx] = child
                        F[rep_idx] = f_child
                    else:
                        sums = np.sum(F, axis=1)
                        worst_idx = np.argmax(sums)
                        X[worst_idx] = child
                        F[worst_idx] = f_child
            else:
                # Dominated: only replace if child has better sum than worst
                sums = np.sum(F, axis=1)
                worst_idx = np.argmax(sums)
                if np.sum(f_child) < sums[worst_idx] - 1e-10:
                    X[worst_idx] = child
                    F[worst_idx] = f_child

        return X
# EVOLVE-BLOCK-END
