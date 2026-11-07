"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4 using front-biased initialization and local refinement."""

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

        # Population size: balance between initial coverage and refinement steps
        pop_size = min(40, budget // 5)
        
        # --- Structural Change: Front-Biased Initialization ---
        # For WFG4, the Pareto front is defined by the first variable x_0.
        # The other variables x_1...x_{11} contribute additive noise that is 
        # typically minimized at the lower bound (0 in standard [0,1] bounds).
        # We initialize the population with points that lie on this "front line".
        
        X = np.zeros((pop_size, dim))
        # Sample x_0 uniformly across its range to cover the trade-off curve
        X[:, 0] = lo[0] + (hi[0] - lo[0]) * np.random.random(pop_size)
        # Set remaining variables to their lower bounds (assumed to be the 
        # minimizer of the non-separable additive terms in WFG4)
        for i in range(1, dim):
            X[:, i] = lo[i]
            
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size

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

            # Select parent: prefer non-dominated points with high crowding distance
            # This ensures we refine diverse parts of the front
            if len(parents_idx) > 1:
                cd = crowding_distance(F, nd_mask)
                # Simple tournament: pick 2, choose the one with higher crowding distance
                c1, c2 = np.random.choice(parents_idx, 2, replace=False)
                if cd[c1] >= cd[c2]:
                    parent_idx = c1
                else:
                    parent_idx = c2
            else:
                parent_idx = parents_idx[0]

            parent = X[parent_idx].copy()
            
            # Adaptive Mutation:
            # 1. Small perturbation on x_0 to explore the front locally
            # 2. Slightly larger perturbation on x_1...x_11 to ensure we find 
            #    the exact minimum of the noise terms (in case lo[i] is not 
            #    perfectly optimal due to specific problem variant details)
            
            progress = used / budget
            # Shrink step size over time for refinement
            sigma_x0 = (1.0 - 0.5 * progress) * 0.05 * (hi[0] - lo[0])
            sigma_rest = (1.0 - 0.5 * progress) * 0.1 * (hi[1:] - lo[1:])
            
            child = parent.copy()
            child[0] += sigma_x0 * np.random.standard_normal()
            child[1:] += sigma_rest * np.random.standard_normal(dim - 1)
            
            # Clip to bounds
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replacement Strategy:
            # Replace the worst dominated point. If all are non-dominated, 
            # replace the one with the lowest crowding distance (least diverse).
            
            if np.any(~nd_mask):
                dom_idx = np.where(~nd_mask)[0]
                worst_idx = dom_idx[np.argmax(F[dom_idx].sum(axis=1))]
            else:
                cd = crowding_distance(F, nd_mask)
                worst_idx = np.argmin(cd)

            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
