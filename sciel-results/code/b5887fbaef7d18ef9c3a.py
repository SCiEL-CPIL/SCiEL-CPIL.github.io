"""Grid-based decomposition with Latin Hypercube Sampling for WFG4 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Grid-based decomposition strategy using Latin Hypercube Sampling.
    
    Partitions the 12-dimensional search space into hyper-rectangular cells,
    generates candidates via LHS for maximal spatial coverage, then refines
    the best non-dominated solutions with local perturbations.
    Uses the global seeded RNG for diversity across seeds.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Use the seeded global RNG for diversity across seeds
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        rng = np.random

        def find_nondominated_indices(F):
            """Return indices of non-dominated points using vectorized comparison."""
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            # F has shape (n, n_obj)
            # For each pair (i,j), check if j dominates i
            # j dominates i if F[j] <= F[i] elementwise and F[j] < F[i] in at least one
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
            return np.where(~dominated)[0]

        # Phase 1: Latin Hypercube Sampling for broad coverage
        # Use about 70% of budget for initial LHS sampling
        n_lhs = max(10, int(budget * 0.7))
        
        # Generate LHS samples: one sample per "row" in each dimension
        # For each dimension, divide [0,1] into n_lhs equal intervals and pick one point per interval
        lhs_samples = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Create n_lhs intervals
            intervals = np.arange(n_lhs) / n_lhs
            # Shuffle to get LHS property
            perm = rng.permutation(n_lhs)
            # Pick a random point within each interval
            points = intervals[perm] + rng.random(n_lhs) / n_lhs
            lhs_samples[:, d] = points
        
        # Map to actual bounds
        X_lhs = lo + (hi - lo) * lhs_samples
        F_lhs = np.array([problem(x) for x in X_lhs])
        used = n_lhs

        # Combine all evaluated points
        X_all = X_lhs.copy()
        F_all = F_lhs.copy()

        # Budget remaining for refinement
        remaining = budget - used
        
        # Get non-dominated solutions from LHS
        nd_idx = find_nondominated_indices(F_all)
        
        if len(nd_idx) == 0:
            # Fallback: use all points
            nd_idx = np.arange(len(F_all))
        
        # Refine non-dominated solutions with local perturbations
        # Distribute remaining budget among non-dominated solutions
        if len(nd_idx) > 0 and remaining > 0:
            # Sort by objective sum (better first) to prioritize good solutions
            sums = np.sum(F_all[nd_idx], axis=1)
            sorted_nd = nd_idx[np.argsort(sums)]
            
            # Allocate more budget to better solutions
            n_nd = len(sorted_nd)
            # Allocate budget proportional to rank (better gets more)
            weights = np.arange(n_nd, 0, -1, dtype=float)
            weights /= weights.sum()
            alloc = (remaining * weights).astype(int)
            # Ensure at least 1 for each if possible
            while alloc.sum() < remaining and n_nd > 0:
                min_idx = np.argmin(alloc)
                alloc[min_idx] += 1
            while alloc.sum() > remaining:
                max_idx = np.argmax(alloc)
                if alloc[max_idx] > 1:
                    alloc[max_idx] -= 1
                else:
                    break
            
            for i, nd_i in enumerate(sorted_nd):
                if alloc[i] <= 0:
                    continue
                base_x = X_all[nd_i].copy()
                base_f = F_all[nd_i].copy()
                
                # Local refinement with decreasing step size
                for k in range(alloc[i]):
                    # Adaptive step size: start larger, get smaller
                    progress = k / max(alloc[i], 1)
                    step_size = 0.1 * (hi - lo) * (1.0 - 0.6 * progress)
                    
                    # Gaussian perturbation
                    perturbation = rng.standard_normal(dim) * step_size
                    new_x = base_x + perturbation
                    new_x = np.clip(new_x, lo, hi)
                    
                    new_f = problem(new_x)
                    used += 1
                    
                    # Check if new point is not dominated by base
                    if not (np.all(base_f <= new_f) and np.any(base_f < new_f)):
                        # If new is better in at least one objective and not worse in all
                        if np.any(new_f < base_f):
                            # Replace base
                            X_all[nd_i] = new_x
                            F_all[nd_i] = new_f
                            base_x = new_x.copy()
                            base_f = new_f.copy()
                        else:
                            # Keep as additional point
                            X_all = np.vstack([X_all, new_x])
                            F_all = np.vstack([F_all, new_f])
                    else:
                        # New point is dominated by base, skip
                        pass

        # Final: return the best non-dominated set
        # Find final non-dominated solutions
        final_nd = find_nondominated_indices(F_all)
        
        # Return all evaluated points (the evaluator records all objective vectors)
        # We just need to return something; the return value is ignored
        return X_all
# EVOLVE-BLOCK-END
