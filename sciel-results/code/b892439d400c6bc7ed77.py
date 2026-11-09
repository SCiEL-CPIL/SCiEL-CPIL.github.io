"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive evolutionary multi-objective search for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # Safety margin before 330s kill
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        pop_size = min(30, self.budget // 3)
        
        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for hypervolume
        all_F = [F.copy()]
        
        # Adaptive mutation scale
        mut_scale = 0.1
        
        def is_non_dominated(F):
            """Return boolean array: True if point is non-dominated."""
            n = len(F)
            non_dom = np.ones(n, dtype=bool)
            for i in range(n):
                if not non_dom[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    # j dominates i if j <= i in all objectives and j < i in at least one
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        non_dom[i] = False
                        break
            return non_dom
        
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            nd_mask = is_non_dominated(F)
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) == 0:
                nd_indices = np.arange(pop_size)
            
            # Select parent from non-dominated set
            parent_idx = nd_indices[rng.integers(len(nd_indices))]
            parent = X[parent_idx]
            
            # Adaptive mutation
            child = parent + mut_scale * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Record this point
            all_F.append(fc.reshape(1, -1))
            
            # Replace worst individual in population
            # Use simple sum-based ranking for speed
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc
            
            # Adapt mutation scale based on progress
            if used % 20 == 0:
                # Reduce mutation scale over time for refinement
                progress = used / self.budget
                mut_scale = 0.1 * (1.0 - 0.5 * progress)
        
        return X
# EVOLVE-BLOCK-END
