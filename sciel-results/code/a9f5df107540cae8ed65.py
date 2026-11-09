"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG7 with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use global numpy random state (seeded by evaluator)
        # This ensures different runs for seed 0 and seed 1
        
        # Initialize population
        pop_size = min(20, self.budget // 4)
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for hypervolume
        all_points = list(zip(X, F))
        
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            non_dominated_mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                if not non_dominated_mask[i]:
                    continue
                for j in range(len(F)):
                    if i == j or not non_dominated_mask[j]:
                        continue
                    # Check if j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        non_dominated_mask[i] = False
                        break
            
            # Select parent from non-dominated points
            nd_indices = np.where(non_dominated_mask)[0]
            if len(nd_indices) == 0:
                nd_indices = np.arange(len(F))
            parent_idx = nd_indices[np.random.randint(len(nd_indices))]
            
            # Create child with mutation
            sigma = 0.1
            child = X[parent_idx] + sigma * (hi - lo) * np.random.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            # Evaluate child
            fc = problem(child)
            used += 1
            all_points.append((child, fc))
            
            # Replace worst point in population
            # Use simple sum-based ranking for replacement
            worst_idx = np.argmax(np.sum(F, axis=1))
            X[worst_idx] = child
            F[worst_idx] = fc
        
        return X
# EVOLVE-BLOCK-END
