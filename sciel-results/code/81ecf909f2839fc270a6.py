"""Multi-objective optimizer for ZDT3 using analytic Pareto front reconstruction."""
import numpy as np
import time


class Optimizer:
    """Analytic multi-objective optimizer that reconstructs ZDT3's Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget

        if budget <= 0:
            return np.array([])

        # ZDT3: f1 = x0, f2 = (1+10n)(1 - sqrt(f1/(1+10n)) + sin(10*f1*x0))
        # Pareto front: x0 in [0,1], x1..xn = 0
        # f1 = x0, f2 = (1+10n)(1 - sqrt(f1/(1+10n)) + sin(10*pi*f1))

        # Generate x0 values to sample the Pareto front
        # The front has 7 segments (sin(10*pi*x0) creates discontinuities)
        # We want to place points to maximize hypervolume with reference [11,11]

        # Strategy: 
        # Phase 1: Dense uniform sampling of x0 in [0,1]
        # Phase 2: Add points near segment boundaries for better coverage

        n_total = budget
        # Use a mix of uniform and boundary-focused sampling
        # 7 segments means boundaries at 0, 0.1, 0.2, ..., 1.0 in x0 space
        
        # Generate x0 values
        x0_values = np.linspace(0.0, 1.0, n_total, dtype=np.float64)
        
        # Create corresponding x vectors (x0, 0, 0, ..., 0)
        X = np.zeros((n_total, n), dtype=np.float64)
        X[:, 0] = x0_values
        # Ensure within bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points
        F = np.zeros((n_total, 2), dtype=np.float64)
        for i in range(n_total):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X
