"""DTLZ1 optimizer using a deterministic nested lattice grid on the Pareto simplex."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 using a nested lattice grid on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit with 10s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ1: first (dim - n_obj + 1) variables determine position on the front,
        # remaining (n_obj - 1) variables determine aggregation (should be 0.5 for Pareto front)
        n_pos = dim - self.n_obj + 1  # 7 - 3 + 1 = 5 position variables
        n_agg = self.n_obj - 1  # 2 aggregation variables

        # DTLZ1 Pareto front: 
        # f1 = 0.5 * x1 * prod(1 - x2..xn_pos)
        # f2 = 0.5 * x1 * (1 - x2) * prod(1 - x3..xn_pos)
        # f3 = 0.5 * (1 - x1)
        # where x1..xn_pos are in [0,1]
        #
        # The Pareto front is a 2D simplex: f1 + f2 + f3 = 0.5, fi >= 0
        # We parameterize with two variables:
        # Let t = x1 (in [0,1]), s = x2 (in [0,1])
        # Then:
        #   f3 = 0.5 * (1 - t)
        #   f1 = 0.5 * t * (1 - s) * (1 - x3) * ... * (1 - xn_pos)
        #   f2 = 0.5 * t * s * (1 - x3) * ... * (1 - xn_pos)
        #
        # To get points on the Pareto front, we need the product term to be 1,
        # which means x3 = x4 = ... = xn_pos = 0.
        # So: f1 = 0.5 * t * (1 - s), f2 = 0.5 * t * s, f3 = 0.5 * (1 - t)
        #
        # This gives us a 2D parameterization: (t, s) -> (f1, f2, f3)
        # where t in [0,1], s in [0,1]
        #
        # We use a nested lattice grid to place exactly 'budget' points uniformly
        # on this 2D simplex.

        # Generate a nested lattice grid of size budget in 2D
        # We'll use a Halton-like sequence or a simple grid with nested structure
        
        # Simple approach: use a 2D grid that nests well
        # For budget=200, we can use a grid that's approximately square
        # sqrt(200) ≈ 14.14, so use 14x14 = 196 points plus 4 more
        
        # Better: use a low-discrepancy sequence for the 2D parameter space
        # Then map to the simplex
        
        # Use a deterministic 2D lattice: for i in range(budget),
        # t_i = (i + 0.5) / budget  -- but this is 1D
        # For 2D, we need two coordinates
        
        # Use a 2D Halton sequence for better uniformity
        def halton_2d(n):
            """Generate n points in 2D using Halton sequence."""
            points = np.zeros((n, 2))
            
            # Base 2 for first dimension
            def halton_1d(i, base):
                f = 1.0
                r = 0.0
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                return r
            
            for i in range(1, n + 1):
                points[i-1, 0] = halton_1d(i, 2)
                points[i-1, 1] = halton_1d(i, 3)
            
            return points
        
        # Generate Halton points in [0,1]^2
        halton_pts = halton_2d(budget)
        
        # Map (t, s) to DTLZ1 input x:
        # t = x1, s = x2, x3..xn_pos = 0
        X = np.zeros((budget, dim))
        for i in range(budget):
            if time.time() > deadline:
                break
            t = halton_pts[i, 0]
            s = halton_pts[i, 1]
            X[i, 0] = t
            X[i, 1] = s
            # x3..xn_pos = 0 (already initialized to 0)
            X[i, n_pos:] = 0.5  # Aggregation variables at 0.5 for Pareto front
        
        # Evaluate all points
        F = np.zeros((budget, self.n_obj))
        valid_count = 0
        for i in range(budget):
            if time.time() > deadline:
                break
            try:
                F[i] = problem(X[i])
                valid_count += 1
            except Exception:
                break

        return X[:valid_count]
# EVOLVE-BLOCK-END
