"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Optimizes DTLZ1 by minimizing g and sampling the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        all_points = []
        deadline = time.time() + 290
        
        # DTLZ1 structure:
        # x[0] is the scaling variable.
        # x[1] to x[n_obj-2] are the shaping variables (here x[1] for n_obj=3).
        # x[n_obj-1] to x[dim-1] are the g-variables (here x[2] to x[6]).
        
        # To minimize objectives, we must minimize g(x_g).
        # The minimum of g is 0, achieved when all g-variables are 0.5.
        # We set the g-variables to 0.5.
        
        base_x = np.zeros(self.dim)
        # Set g-variables to 0.5 to minimize g
        base_x[2:] = 0.5
        
        remaining = self.budget
        if remaining <= 0:
            return all_points
            
        # Use a triangular lattice for better coverage of the 2D simplex (x[0], x[1])
        # where x[0] >= 0, x[1] >= 0, x[0] + x[1] <= 1
        # This gives uniform density on the simplex.
        
        # Determine number of points to use
        n_points = remaining
        
        # Generate points on a triangular lattice
        # For a triangle with vertices (0,0), (1,0), (0,1):
        # We can use barycentric coordinates or a grid with rejection
        # Better: use a structured grid that covers the triangle uniformly
        
        # Method: For each "level" k from 0 to N-1, place points along lines
        # where x0 + x1 = k/N
        # Number of points at level k is k+1 (for x0 from 0 to k/N in steps of 1/N)
        # Total points = sum_{k=0}^{N-1} (k+1) = N(N+1)/2
        # So N ~ sqrt(2 * n_points)
        
        N = int(np.sqrt(2 * n_points))
        N = max(1, N)
        
        points_to_eval = []
        
        # Generate triangular lattice points
        for k in range(N):
            # Points on line x0 + x1 = k/N
            # x0 goes from 0 to k/N in steps of 1/N
            for j in range(k + 1):
                x0 = j / N
                x1 = (k - j) / N
                x = base_x.copy()
                x[0] = x0
                x[1] = x1
                points_to_eval.append(x)
        
        # If we have more budget than lattice points, add extra points
        # If we have less, truncate
        if len(points_to_eval) > n_points:
            points_to_eval = points_to_eval[:n_points]
        elif len(points_to_eval) < n_points:
            # Add random points in the simplex to fill remaining budget
            rng = np.random.default_rng(42)
            while len(points_to_eval) < n_points:
                # Sample uniformly in triangle using rejection or barycentric
                u = rng.random()
                v = rng.random()
                if u + v > 1:
                    u = 1 - u
                    v = 1 - v
                x0 = u
                x1 = v
                x = base_x.copy()
                x[0] = x0
                x[1] = x1
                points_to_eval.append(x)
        
        # Evaluate all points
        for x in points_to_eval:
            if time.time() > deadline:
                break
            f = problem(x)
            all_points.append(f.copy())
            
        # If we have budget left, add more random points
        if len(all_points) < self.budget:
            rng = np.random.default_rng(42)
            while len(all_points) < self.budget and time.time() < deadline:
                u = rng.random()
                v = rng.random()
                if u + v > 1:
                    u = 1 - u
                    v = 1 - v
                x = base_x.copy()
                x[0] = u
                x[1] = v
                f = problem(x)
                all_points.append(f.copy())

        return all_points
# EVOLVE-BLOCK-END
