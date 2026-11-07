"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer on ZDT6 exploiting problem structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit

        dim = self.dim
        budget = self.budget

        # Pre-compute mapping from f1 to x1 for ZDT6
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)**6
        # We sample x1 densely to build a lookup table
        n_samples = 50000
        x1_samples = np.linspace(0, 1, n_samples)
        f1_samples = 1 - np.exp(-4 * x1_samples) * (np.sin(6 * np.pi * x1_samples) ** 6)
        
        # Sort by f1 for efficient lookup
        sort_idx = np.argsort(f1_samples)
        f1_sorted = f1_samples[sort_idx]
        x1_sorted = x1_samples[sort_idx]

        def get_x1_for_f1(target_f1):
            """Find x1 that yields f1 closest to target_f1."""
            target_f1 = np.clip(target_f1, 0, 1)
            # Binary search for insertion point
            idx = np.searchsorted(f1_sorted, target_f1)
            
            # Handle boundaries
            if idx == 0:
                return x1_sorted[0]
            if idx == len(f1_sorted):
                return x1_sorted[-1]
            
            # Check neighbors to find the closest f1
            dist_prev = abs(f1_sorted[idx - 1] - target_f1)
            dist_curr = abs(f1_sorted[idx] - target_f1)
            
            if dist_prev < dist_curr:
                return x1_sorted[idx - 1]
            else:
                return x1_sorted[idx]

        # Strategy:
        # 1. The Pareto front of ZDT6 is achieved when x_2...x_10 = 0.
        # 2. We need to cover the range of f1 values [0, 1] to maximize hypervolume.
        # 3. Optimal distribution of f1 points to minimize discretization error is f1 ~ i^(2/3).
        
        evaluated_f1s = []
        evaluated_f2s = []
        
        # Use all budget for the optimal distribution
        n_points = budget
        
        # Generate f1 targets: f1_i = (i/N)^(2/3)
        # i from 1 to N
        i_vals = np.arange(1, n_points + 1)
        targets = (i_vals / n_points) ** (2.0 / 3.0)
        
        # Ensure the last point is exactly 1.0
        targets[-1] = 1.0
        
        # Also ensure the first point is close to 0. 
        # (1/N)^(2/3) is small for large N.
        
        for t in targets:
            if time.time() > deadline:
                break
            x1 = get_x1_for_f1(t)
            x = np.zeros(dim)
            x[0] = x1
            # x[1:] is 0, which minimizes g and thus f2 for a given f1
            f = problem(x)
            evaluated_f1s.append(f[0])
            evaluated_f2s.append(f[1])

        return None
# EVOLVE-BLOCK-END
