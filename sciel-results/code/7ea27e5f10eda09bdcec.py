"""Differential Evolution multi-objective optimizer with time-aware budget management."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective optimizer with strict time and budget constraints."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.pop_size = 50  # Fixed reasonable population size for DE

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Initialize population
        X = lo + span * np.random.rand(self.pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        used = self.pop_size
        
        # DE Parameters
        CR = 0.9
        F_factor = 0.8
        
        # Main Loop
        while used < self.budget:
            # Check time budget
            if time.time() > deadline:
                break
            
            # Generate offspring
            child = np.zeros(self.dim)
            f_child = None
            
            # DE/rand/1/bin mutation
            i = np.random.randint(self.pop_size)
            a, b, c = np.random.choice(self.pop_size, 3, replace=False)
            
            # Ensure distinct indices for mutation
            while a == b or a == c or b == c:
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
            
            v = X[a] + F_factor * (X[b] - X[c])
            
            # Crossover
            j_rand = np.random.randint(self.dim)
            for j in range(self.dim):
                if np.random.rand() <= CR or j == j_rand:
                    child[j] = v[j]
                else:
                    child[j] = X[i][j]
            
            # Clip to bounds
            child = np.clip(child, lo, hi)
            
            # Evaluate
            f_child = problem(child)
            used += 1
            
            # Replacement: Simple non-dominated check against parent
            # If child dominates parent, replace
            # If parent dominates child, discard
            # If non-dominated, keep both (but we have fixed pop size, so we need a strategy)
            # For simplicity and speed, we use a "tournaments" or simple replacement.
            # Let's use a simple approach: If child is better than the worst point in the population
            # or dominates the parent, replace.
            
            # Check if child dominates parent
            dominates_parent = np.all(f_child <= F[i]) and np.any(f_child < F[i])
            
            if dominates_parent:
                X[i] = child
                F[i] = f_child
            else:
                # Check if parent dominates child
                dominates_child = np.all(F[i] <= f_child) and np.any(F[i] < f_child)
                if not dominates_child:
                    # Non-dominated. 
                    # We can replace the parent if the child has a better sum of objectives (simple heuristic)
                    # Or just keep the population diverse.
                    # Let's try: if child sum < parent sum, replace.
                    if np.sum(f_child) < np.sum(F[i]):
                        X[i] = child
                        F[i] = f_child
                    else:
                        # Occasionally replace to maintain diversity
                        if np.random.rand() < 0.1:
                            X[i] = child
                            F[i] = f_child

        return X
# EVOLVE-BLOCK-END
