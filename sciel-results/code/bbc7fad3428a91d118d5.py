# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points in F (n x n_obj)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        
        # F is (n, n_obj)
        # A point i is dominated if there exists j such that F[j] <= F[i] and F[j] != F[i]
        # Vectorized check:
        # For each i, check if any j dominates i.
        # F[j] <= F[i] for all objectives.
        
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            # Check all j != i
            # F[j] <= F[i]
            le = np.all(F <= F[i], axis=1)
            lt = np.any(F < F[i], axis=1)
            # Exclude i itself
            le[i] = False
            if np.any(le & lt):
                dominated[i] = True
        return np.where(~dominated)[0]

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)
        
        # Population storage
        # We maintain a fixed size population to keep non-domination check O(N^2) cheap
        # N=30 is 900 ops per check, very fast.
        pop_size = 30
        X = []
        F = []
        used = 0
        
        def eval_point(x):
            nonlocal used
            if used >= self.budget or time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            X.append(x)
            F.append(f)
            used += 1
            return f

        # Phase 1: Informed Initialization
        # WFG7 is multi-modal. The Pareto front consists of points where each x_i
        # is at a specific "peak" of the oscillatory functions.
        # In [0,1] space, common peaks for WFG terms are around 0.35, 0.5, 0.65 etc.
        # Let's sample from a discrete set of "candidate" values for each dimension
        # mixed with random noise to find the front quickly.
        
        # Candidate values based on typical WFG peak locations in normalized space
        # The exact location depends on the specific WFG7 definition used in the evaluator,
        # but 0.35 and 0.65 are often the "good" modes for sin(2*pi*x) type terms 
        # when combined with the translation/scaling.
        # We will use a set of values to initialize.
        peak_vals = [0.35, 0.5, 0.65, 0.2, 0.8, 0.4, 0.6]
        
        # Generate initial population
        n_init = min(pop_size, self.budget // 2)
        for i in range(n_init):
            if time.time() > deadline:
                break
            if i < pop_size // 2:
                # Half random
                x = lo + (hi - lo) * rng.random(self.dim)
            else:
                # Half "peak-biased": each dimension picks a peak value + small noise
                x = np.array([
                    lo[d] + (hi[d] - lo[d]) * (peak_vals[rng.integers(len(peak_vals))] + 0.1 * rng.standard_normal())
                    for d in range(self.dim)
                ])
                x = np.clip(x, lo, hi)
            
            f = eval_point(x)
            if f is None:
                break

        if not X:
            return X

        # Phase 2: Evolutionary Loop
        # Maintain population size. If we exceed, remove worst.
        # "Worst" is defined by highest sum of objectives (simple scalarization)
        # or highest distance to reference. Sum is faster.
        
        # Adaptive mutation scale
        # Start large, shrink to refine
        base_step = 0.2
        shrink_rate = 0.95
        
        while used < self.budget and time.time() < deadline:
            F_arr = np.array(F)
            X_arr = np.array(X)
            
            # Find non-dominated
            nd_idx = self._non_dominated_indices(F_arr)
            
            # Select parent
            if len(nd_idx) > 0:
                # Select from non-dominated
                parent_idx = nd_idx[rng.integers(len(nd_idx))]
            else:
                # Fallback: select from all
                parent_idx = rng.integers(len(F))
            
            parent = X_arr[parent_idx].copy()
            
            # Mutate
            # Scale step by remaining budget to ensure refinement at the end
            remaining_frac = max(0.1, (self.budget - used) / self.budget)
            step = base_step * remaining_frac
            # Add some randomness to step size
            step *= (0.5 + rng.random())
            
            mutation = step * (hi - lo) * rng.standard_normal(self.dim)
            child = parent + mutation
            child = np.clip(child, lo, hi)
            
            fc = eval_point(child)
            if fc is None:
                break
            
            # Maintain population size
            if len(X) > pop_size:
                # Remove the worst point
                # We want to keep points that are "good" (low objectives)
                # Use sum of objectives as a proxy
                F_arr = np.array(F)
                sums = np.sum(F_arr, axis=1)
                # We might want to be careful not to remove a non-dominated point
                # if a dominated point has a lower sum? 
                # Actually, non-dominated points are generally "better".
                # Let's prioritize removing dominated points with high sums.
                # If all are non-dominated, remove the one with highest sum.
                
                # Identify dominated points
                dominated_mask = np.zeros(len(F), dtype=bool)
                # Re-calculate domination status for removal decision
                # This is O(N^2) again, but N=30 is fine.
                # Optimization: We already have nd_idx. 
                # Points not in nd_idx are dominated.
                dominated_idx = np.setdiff1d(np.arange(len(F)), nd_idx)
                
                if len(dominated_idx) > 0:
                    # Among dominated, remove the one with highest sum
                    dom_sums = sums[dominated_idx]
                    worst_dom_local = np.argmax(dom_sums)
                    worst_idx = dominated_idx[worst_dom_local]
                else:
                    # All non-dominated, remove highest sum
                    worst_idx = np.argmax(sums)
                
                X.pop(worst_idx)
                F.pop(worst_idx)
            
            # Shrink step
            base_step *= shrink_rate
            
            # Optional: If base_step becomes too small, restart with a random jump?
            # For 200 budget, this might not be necessary.
        
        return X
# EVOLVE-BLOCK-END
