"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer combining simple ES with NSGA-II-style selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use global numpy random to respect external seed
        # Initial population: mix of random and Pareto-front-biased samples
        pop_size = min(40, budget // 5)
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        
        # WFG4 Pareto front bias: 
        # WFG4 is a shifted and rotated version of WFG1.
        # The Pareto front of WFG1 (3 objectives) is a simplex in the first 3 dimensions.
        # Specifically, for WFG1, the Pareto front is defined by f1+x1 = 1, f2+x2 = 1, f3+x3 = 1?
        # Actually, WFG1 has a simplex front. 
        # A common heuristic for WFG problems is that the Pareto front lies in a lower-dimensional manifold.
        # For WFG4, the front is known to be a simplex in the objective space.
        # Let's try sampling points that are likely on or near the front.
        # In WFG4, the first few variables often control the position on the front.
        # We will add some samples biased towards the center of the domain or specific structures.
        
        # Add a few samples near the center, which is often a good starting point for WFG
        center = (lo + hi) / 2
        for _ in range(5):
            if used >= budget: break
            x = center + 0.05 * (hi - lo) * np.random.standard_normal(dim)
            x = np.clip(x, lo, hi)
            X = np.vstack([X, x])
        
        F = np.array([problem(x) for x in X])
        used = len(X)

        def non_dominated_mask(F):
            """Return boolean mask of non-dominated points in F."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                diff = F - F[i]
                dominates_i = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
                dominates_i[i] = False
                if np.any(dominates_i):
                    mask[i] = False
            return mask

        def crowding_distance(F, mask):
            """Compute crowding distance for non-dominated points."""
            n = len(F)
            dist = np.zeros(n)
            if not np.any(mask) or np.sum(mask) <= 2:
                return dist
            idx = np.where(mask)[0]
            for m in range(n_obj):
                order = idx[np.argsort(F[idx, m])]
                if len(order) > 2:
                    dist[order[0]] = np.inf
                    dist[order[-1]] = np.inf
                    f_range = F[order[-1], m] - F[order[0], m]
                    if f_range > 1e-12:
                        for i in range(1, len(order) - 1):
                            dist[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / f_range
            return dist

        while used < budget:
            if time.time() > deadline:
                break

            # Find non-dominated front
            nd_mask = non_dominated_mask(F)
            nd_count = np.sum(nd_mask)

            if nd_count == 0:
                parents_idx = np.arange(len(F))
            else:
                parents_idx = np.where(nd_mask)[0]

            # Select parent via tournament on non-dominated + crowding
            if len(parents_idx) > 1:
                cd = crowding_distance(F, nd_mask)
                c1, c2 = np.random.choice(parents_idx, 2, replace=False)
                if cd[c1] >= cd[c2]:
                    parent = X[c1]
                else:
                    parent = X[c2]
            else:
                parent = X[parents_idx[0]]

            # Adaptive mutation: start large, shrink over time
            progress = used / budget
            sigma = (1.0 - 0.7 * progress) * 0.3 * (hi - lo)
            child = parent + sigma * np.random.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst point
            dom_mask = ~nd_mask
            if np.any(dom_mask):
                dom_idx = np.where(dom_mask)[0]
                worst_idx = dom_idx[np.argmax(F[dom_idx].sum(axis=1))]
            else:
                cd = crowding_distance(F, nd_mask)
                worst_idx = np.argmin(cd)

            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
