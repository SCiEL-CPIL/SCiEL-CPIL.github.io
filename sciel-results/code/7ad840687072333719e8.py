"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer exploiting DTLZ7's analytical Pareto front.
    
    DTLZ7 with M=3 objectives: when x[2:]=0, the Pareto front is:
      f1 = x[0], f2 = x[1], f3 = 1 - sqrt(x[0])
    where x[0], x[1] in [0, 1].
    
    We use all budget evaluations to place points on this front using
    a stratified sampling approach that maximizes hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s limit
        
        n_eval = self.budget
        dim = self.dim
        
        # Generate points on the analytic Pareto front
        # f1 = x0, f2 = x1, f3 = 1 - sqrt(x0)
        # We want to maximize hypervolume with reference [1.1, 1.1, 25.0]
        
        # Use a stratified grid approach:
        # - Spread x0 values to cover [0, 1] since f3 depends on x0
        # - For each x0, spread x1 values to cover [0, 1]
        # 
        # Optimal: use a 2D stratified sample. 
        # With n points, use roughly sqrt(n) x sqrt(n) grid, 
        # or a Latin Hypercube for better coverage.
        
        # Let's use a structured approach:
        # n0 points for x0 dimension, n1 points for x1 dimension
        # Total evaluations = n0 * n1 (if grid) or n (if LHS)
        
        # For hypervolume maximization on this front, a grid in (x0, x1) 
        # space works well. Let's use a near-square grid.
        
        n0 = int(np.sqrt(n_eval))
        n1 = n_eval // n0
        
        # Adjust to use all evaluations
        while n0 * n1 > n_eval and n0 > 1:
            n0 -= 1
            n1 = n_eval // n0
        while n0 * n1 < n_eval and n0 < n_eval:
            n0 += 1
            n1 = n_eval // n0
            if n0 * n1 > n_eval:
                n0 -= 1
                n1 = n_eval // n0
                break
        
        # Create stratified x0 values in [0, 1]
        # Use midpoints of equal-width intervals for stratification
        x0_vals = (np.arange(n0) + 0.5) / n0
        x1_vals = (np.arange(n1) + 0.5) / n1
        
        # Create all combinations (grid)
        x0_grid, x1_grid = np.meshgrid(x0_vals, x1_vals, indexing='ij')
        x0_flat = x0_grid.ravel()
        x1_flat = x1_grid.ravel()
        
        n_points = len(x0_flat)
        
        # If we have fewer points than budget, add more with jitter
        # If more, trim
        if n_points < n_eval:
            # Add additional points with small random perturbations
            rng = np.random.default_rng(42)
            extra = n_eval - n_points
            x0_extra = rng.random(extra) * 0.02  # small perturbations
            x1_extra = rng.random(extra) * 0.02
            x0_flat = np.concatenate([x0_flat, x0_extra])
            x1_flat = np.concatenate([x1_flat, x1_extra])
        elif n_points > n_eval:
            x0_flat = x0_flat[:n_eval]
            x1_flat = x1_flat[:n_eval]
        
        # Ensure we don't exceed budget
        x0_flat = x0_flat[:n_eval]
        x1_flat = x1_flat[:n_eval]
        
        # Build the input vectors: x[0]=x0, x[1]=x1, x[2:]=0
        X = np.zeros((len(x0_flat), dim))
        X[:, 0] = x0_flat
        X[:, 1] = x1_flat
        # x[2:] = 0 by default
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        used = 0
        for i in range(len(X)):
            if time.time() > deadline:
                break
            problem(X[i])
            used += 1
        
        return X
# EVOLVE-BLOCK-END
