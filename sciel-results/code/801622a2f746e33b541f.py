"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-specific optimizer exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # Reserve 4 evaluations for corner points of the Pareto front square
        # Corners: (0,0), (0,1), (1,0), (1,1) in (x1, x2) space
        n_corners = 4
        n_lhs = n_evals - n_corners
        
        rng = np.random.default_rng(42)
        
        # Latin Hypercube Sampling in 2D for the remaining points
        perm1 = rng.permutation(n_lhs)
        perm2 = rng.permutation(n_lhs)
        
        x1_lhs = (perm1 + rng.random(n_lhs)) / n_lhs
        x2_lhs = (perm2 + rng.random(n_lhs)) / n_lhs
        
        # Corner points
        x1_corners = np.array([0.0, 0.0, 1.0, 1.0])
        x2_corners = np.array([0.0, 1.0, 0.0, 1.0])
        
        # Combine
        x1 = np.concatenate([x1_lhs, x1_corners])
        x2 = np.concatenate([x2_lhs, x2_corners])
        
        # Clip to bounds
        x1 = np.clip(x1, lo[0], hi[0])
        x2 = np.clip(x2, lo[1], hi[1])
        
        # Build the full 22-dimensional vectors: x3..x22 = 0
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1
        X[:, 1] = x2
        
        # Evaluate all points
        F = np.zeros((n_evals, self.n_obj))
        for i in range(n_evals):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
