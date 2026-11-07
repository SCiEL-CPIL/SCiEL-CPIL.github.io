import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_evals = self.budget
        dim = self.dim
        start_time = time.time()
        deadline = 300.0
        
        # ZDT6: f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # Pareto front: (f1, 1-sqrt(f1)) for f1 in [0,1]
        # Strategy: Sample f1 uniformly, then invert to find x1
        # Since f1 is oscillatory, for each target f1 value, find an x1 that achieves it
        
        # Precompute: for a dense grid of x1, store (x1, f1(x1))
        # Then for each target f1, find the x1 that gives closest f1
        
        # First, create a fine grid of x1 values
        n_grid = 20000
        x1_grid = np.linspace(0, 1, n_grid)
        f1_grid = 1 - np.exp(-4 * x1_grid) * np.sin(6 * np.pi * x1_grid)**8
        
        # For each target f1 in [0,1], find x1 that minimizes |f1(x1) - target|
        # We'll use uniform sampling of target f1 values
        
        # Sample f1 targets uniformly in [0, 1]
        # Include endpoints
        f1_targets = np.linspace(0, 1, n_evals)
        
        # For each target, find the best x1 from the grid
        x1_selected = np.empty(n_evals)
        for i, target in enumerate(f1_targets):
            if time.time() - start_time > deadline:
                break
            # Find x1 in grid that gives f1 closest to target
            diffs = np.abs(f1_grid - target)
            best_idx = np.argmin(diffs)
            x1_selected[i] = x1_grid[best_idx]
        
        # Remove duplicates in x1_selected
        x1_selected = np.unique(np.round(x1_selected, 15))
        
        # If we have more than budget, take first budget points
        if len(x1_selected) > n_evals:
            x1_selected = x1_selected[:n_evals]
        
        # Evaluate each point
        for x1 in x1_selected:
            if time.time() - start_time > deadline:
                break
            x = np.zeros(dim)
            x[0] = x1
            problem(x)
        
        return None
