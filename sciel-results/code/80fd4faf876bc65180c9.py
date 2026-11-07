"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer on ZDT6 using a structured search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # ZDT6 specific initialization
        # x0 controls the shape of the Pareto front (0 to 1)
        # x1..x9 affect the second objective (distance from front)
        # Good initialization: x0 spread out, x1..x9 small
        
        pop_size = 30
        X = np.zeros((pop_size, self.dim))
        # Spread x0 uniformly
        X[:, 0] = rng.uniform(0, 1, pop_size)
        # Set x1..x9 to small values (close to front)
        X[:, 1:] = rng.uniform(0, 0.1, (pop_size, self.dim - 1))
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: non-dominated sort (simplified: just find non-dominated set)
        def get_non_dominated(F):
            mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                if not mask[i]:
                    continue
                # Check if any other point dominates i
                # Dominates: F[j] <= F[i] and F[j] < F[i] in at least one dim
                dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
                dom[i] = False  # exclude self
                if np.any(dom):
                    mask[i] = False
            return mask
        
        # Helper: crowding distance (simplified)
        def crowding_distance(F, mask):
            dist = np.zeros(len(F))
            indices = np.where(mask)[0]
            n = len(indices)
            if n <= 2:
                dist[indices] = np.inf
                return dist
            
            for m in range(self.n_obj):
                sorted_idx = indices[np.argsort(F[indices, m])]
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_min = F[sorted_idx[0], m]
                f_max = F[sorted_idx[-1], m]
                if f_max == f_min:
                    continue
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (F[sorted_idx[i+1], m] - F[sorted_idx[i-1], m]) / (f_max - f_min)
            
            return dist
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            nd_mask = get_non_dominated(F)
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) == 0:
                # Fallback: pick random
                parent_idx = rng.integers(len(F))
            else:
                # Use crowding distance to select parent
                cd = crowding_distance(F, nd_mask)
                # Select from non-dominated with probability proportional to crowding distance
                probs = cd[nd_indices]
                probs = probs / probs.sum()
                parent_idx = nd_indices[rng.choice(len(nd_indices), p=probs)]
            
            # Mutation: Gaussian with adaptive step size
            # Step size decreases over time
            progress = used / self.budget
            step_size = 0.1 * (1.0 - 0.5 * progress)
            
            # For ZDT6, x0 should change slowly to cover the front
            # x1..x9 should be kept small
            child = X[parent_idx].copy()
            child += step_size * range_ * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            # Special handling: keep x1..x9 small
            # With some probability, reset x1..x9 to small values
            if rng.random() < 0.3:
                child[1:] = rng.uniform(0, 0.05, self.dim - 1)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point (by sum of objectives, among dominated if possible)
            # Find dominated points
            dom_mask = ~nd_mask
            if np.any(dom_mask):
                # Replace the worst dominated point
                worst_indices = np.where(dom_mask)[0]
                worst_idx = worst_indices[np.argmax(F[worst_indices, 0] + F[worst_indices, 1])]
            else:
                # All non-dominated, replace the one with largest sum
                worst_idx = np.argmax(F[:, 0] + F[:, 1])
            
            X[worst_idx] = child
            F[worst_idx] = fc
        
        return X
# EVOLVE-BLOCK-END
