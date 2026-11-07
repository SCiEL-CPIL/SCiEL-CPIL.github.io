"""Improved program using Differential Evolution (DE) for multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Multi-objective optimizer using Differential Evolution."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.evaluations = 0

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # 200 budget. 40 initial + 160 offspring.
        N = 40
        
        rng = np.random.default_rng(42)
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        self.evaluations += N
        
        # DE parameters
        F_factor = 0.8  # Scale factor
        
        # Main loop
        while self.evaluations < self.budget:
            # Select random index for target
            i = rng.integers(0, N)
            
            # Select two other distinct indices
            j = rng.integers(0, N)
            k = rng.integers(0, N)
            while j == i or k == i or j == k:
                j = rng.integers(0, N)
                k = rng.integers(0, N)
            
            # Create mutant vector: v = x_j + F * (x_k - x_i)
            v = X[j] + F_factor * (X[k] - X[i])
            v = np.clip(v, lo, hi)
            
            # Evaluate mutant
            f_v = problem(v)
            self.evaluations += 1
            
            # Compare f_v and F[i]
            f_i = F[i]
            
            # Check if v dominates i
            if np.all(f_v <= f_i) and np.any(f_v < f_i):
                X[i] = v
                F[i] = f_v
            # Check if i dominates v
            elif np.all(f_i <= f_v) and np.any(f_i < f_v):
                pass # Keep i
            else:
                # Non-dominated. Replace with some probability to maintain diversity
                # Using 0.5 is a standard heuristic
                if rng.random() < 0.5:
                    X[i] = v
                    F[i] = f_v

        return X
# EVOLVE-BLOCK-END
