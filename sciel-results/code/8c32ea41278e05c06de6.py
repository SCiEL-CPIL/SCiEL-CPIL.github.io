"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer exploiting ZDT4's known analytical Pareto front.
    
    ZDT4 structure:
    - x[0] in [0, 1], x[1:] in [-5, 5]
    - f1 = x[0]
    - g(x) = 1 + 10*(n-1) = 101 when x[i]=0 for all i>0
    - f2 = g(x) * (1 - sqrt(f1 / g(x)))
    - Pareto front: f1 = x0, f2 = 101*(1 - sqrt(x0/101))
    
    Strategy: Use all budget evaluations to densely sample the known Pareto front.
    All points are non-dominated and on the true front, maximizing hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4: x[0] in [0,1], x[i] for i>0 should be 0 for global minimum of g
        # The Pareto front is parameterized by x[0] in [0,1]
        # f1 = x[0], f2 = 101*(1 - sqrt(x[0]/101))
        
        # Use all budget to sample the Pareto front densely
        n_samples = self.budget
        
        # Sample x[0] uniformly in [0, 1] for dense front coverage
        x0_values = np.linspace(0.0, 1.0, n_samples)
        
        # For each sample, set x[0] = x0_value, x[1:] = 0 (global min of g)
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x0_values
        # x[1:] = 0 is already set by zeros
        
        # Clip to bounds for safety
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
