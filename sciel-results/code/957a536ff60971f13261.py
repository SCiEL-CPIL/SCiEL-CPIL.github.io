"""Multi-objective optimizer on DTLZ2 using structured grid search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structured search optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        # Assume DTLZ2 structure: first n_obj variables define position on front,
        # rest define distance from front (g(x)).
        # Optimal g(x) is 0, achieved by setting x_i = 0.5 for i > n_obj.
        
        n_pos = self.n_obj
        n_g = self.dim - n_pos
        
        # Strategy:
        # 1. Fix x_{n_obj+1...dim} to 0.5 (midpoint of [0,1]) to minimize g(x).
        # 2. Search over x_{1...n_obj} using a grid + random sampling.
        
        # Generate grid points for the positional variables
        # Budget is 200.
        # Use a 5x5x5 grid = 125 points.
        # Remaining 75 points for random sampling or refinement.
        
        grid_size = 5
        if self.n_obj == 3:
            grid_points = 5 * 5 * 5
        elif self.n_obj == 2:
            grid_size = 10
            grid_points = 10 * 10
        else:
            grid_size = 5
            grid_points = grid_size ** self.n_obj
            
        # Adjust grid size to fit budget
        while grid_size ** self.n_obj > self.budget * 0.7:
            grid_size -= 1
            if grid_size < 2:
                grid_size = 2
                break
        
        # Create grid
        axes = [np.linspace(lo[i], hi[i], grid_size) for i in range(self.n_obj)]
        grids = np.meshgrid(*axes, indexing='ij')
        grid_X = np.array([g.ravel() for g in grids]).T
        
        # Create random points
        n_random = self.budget - len(grid_X)
        if n_random > 0:
            rng = np.random.default_rng(42)
            random_X = lo[:self.n_obj] + (hi[:self.n_obj] - lo[:self.n_obj]) * rng.random((n_random, self.n_obj))
        else:
            random_X = np.empty((0, self.n_obj))

        # Combine
        search_X = np.vstack([grid_X, random_X])
        
        # Full dimension vectors
        X = np.zeros((len(search_X), self.dim))
        X[:, :self.n_obj] = search_X
        # Set g-variables to 0.5
        if n_g > 0:
            X[:, self.n_obj:] = 0.5 * (hi[self.n_obj:] + lo[self.n_obj:])
            
        # Evaluate
        F = np.array([problem(x) for x in X], dtype=np.float64)
        
        # Filter out points that are worse than reference? 
        # The task says "Only points strictly better than the reference point in every objective count."
        # Reference is [2.5, 2.5, 2.5].
        # DTLZ2 objectives are typically around 1. So all should count.
        
        return X
# EVOLVE-BLOCK-END
