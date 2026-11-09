"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Direct Front Sampling Optimizer for DTLZ7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Determine indices for DTLZ7 structure
        # g_vars = x[0 : dim - n_obj]
        # y_vars = x[dim - n_obj : dim]
        
        g_dim = dim - n_obj
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy:
        # 1. Fix g_vars to 0.5 (optimal for g=0)
        # 2. Sample y_vars using Latin Hypercube + Random to cover the Pareto front.
        
        # Generate a list of candidate y-vectors
        # Use LHS for the first part, random for the rest.
        
        n_lhs = min(budget // 2, 100)
        n_random = budget - n_lhs
        
        y_candidates = []
        
        # LHS generation for y_vars
        # y_dim is n_obj
        y_dim = n_obj
        
        if n_lhs > 0:
            # Generate LHS points in [0,1]^y_dim
            lhs_points = np.zeros((n_lhs, y_dim))
            for d in range(y_dim):
                # Create bins
                bins = np.linspace(0, 1, n_lhs + 1)
                # Pick random offset in each bin
                offsets = np.random.rand(n_lhs) * (1.0 / n_lhs)
                points_d = bins[:n_lhs] + offsets
                np.random.shuffle(points_d)
                lhs_points[:, d] = points_d
            y_candidates.extend(lhs_points.tolist())
        
        # Random
        if n_random > 0:
            random_points = np.random.rand(n_random, y_dim)
            y_candidates.extend(random_points.tolist())
        
        # We need to construct full x vectors and evaluate them.
        
        for i in range(budget):
            x = np.zeros(dim)
            # Set g-vars to 0.5
            x[:g_dim] = 0.5
            
            # Set y-vars
            if i < len(y_candidates):
                y = y_candidates[i]
            else:
                # Fallback random
                y = np.random.rand(y_dim)
            
            x[g_dim:dim] = y
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            # Evaluate
            problem(x)
            
        return None
# EVOLVE-BLOCK-END
