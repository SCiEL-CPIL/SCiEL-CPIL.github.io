"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer with deadline guard and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng_size = hi - lo
        
        # Initial population
        pop_size = min(30, self.budget // 3)
        X = lo + rng_size * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size
        
        # Track best solutions
        all_F = F.copy()
        all_X = X.copy()
        
        def is_dominated(F_arr):
            """Return boolean array: True if dominated by another in F_arr."""
            n = len(F_arr)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
            return dominated
        
        def crowding_distance(F_arr):
            """Compute crowding distance for diversity maintenance."""
            n = len(F_arr)
            if n <= 2:
                return np.ones(n)
            
            # Normalize objectives
            F_min = F_arr.min(axis=0)
            F_max = F_arr.max(axis=0)
            F_range = F_max - F_min
            F_range[F_range == 0] = 1.0
            F_norm = (F_arr - F_min) / F_range
            
            cd = np.zeros(n)
            for obj in range(self.n_obj):
                order = np.argsort(F_norm[:, obj])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[order[i]] += (F_norm[order[i+1], obj] - F_norm[order[i-1], obj])
            
            # Handle infinite values
            cd[cd == np.inf] = cd.max() if np.any(cd > 0) else 1.0
            return cd
        
        # Main evolution loop
        mutation_scale = 0.15
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Find non-dominated set
            dominated = is_dominated(F)
            non_dom_idx = np.where(~dominated)[0]
            
            if len(non_dom_idx) == 0:
                # All dominated, pick random
                non_dom_idx = np.array([rng.integers(len(F))])
            
            # Select parent with crowding distance weighting
            if len(non_dom_idx) > 1:
                cd = crowding_distance(F[non_dom_idx])
                probs = cd / cd.sum()
                parent_idx = non_dom_idx[rng.choice(len(non_dom_idx), p=probs)]
            else:
                parent_idx = non_dom_idx[0]
            
            # Adaptive mutation
            progress = used / self.budget
            current_scale = mutation_scale * (1.0 - 0.7 * progress)
            
            child = X[parent_idx] + current_scale * rng_size * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Update all solutions
            all_F = np.vstack([all_F, fc.reshape(1, -1)])
            all_X = np.vstack([all_X, child.reshape(1, -1)])
            
            # Update population: replace worst if child is better
            # Use hypervolume-like score: sum of normalized objectives
            def hv_score(f):
                # Reference point for scoring
                ref = np.array([3000.0, 0.06])
                if np.any(f >= ref):
                    return 0.0
                # Simple sum-based score (lower is better)
                return -np.sum(f)
            
            # Find worst in current population
            scores = np.array([hv_score(f) for f in F])
            worst_idx = np.argmin(scores)  # Most negative = worst
            
            # Compare child with worst
            if hv_score(fc) > scores[worst_idx]:
                X[worst_idx] = child
                F[worst_idx] = fc
            elif rng.random() < 0.1:
                # Occasionally replace even if worse, for diversity
                X[worst_idx] = child
                F[worst_idx] = fc
        
        return all_X
# EVOLVE-BLOCK-END
