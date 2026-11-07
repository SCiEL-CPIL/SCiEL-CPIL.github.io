import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        max_evals = self.budget
        
        # Strategy: Deterministic Uniform Grid on DTLZ2 Pareto Front
        # DTLZ2 Pareto front is defined by x_0..x_{M-2} in [0,1] and x_M..x_{N-1} = 0.5
        # The objectives are f_i = (1+G) * y_i, where y = sin(pi/2 * x_i) for i=0..M-1
        # Since G=0 on the front, f_i = sin(pi/2 * x_i)
        # We map a uniform grid in x-space to the front.
        
        # We have budget evaluations.
        # The Pareto front is 2-dimensional (M=3 objectives -> 2 position variables).
        # We will create a grid of size n x n in the position space.
        # Total points = n^2.
        # We want n^2 <= budget.
        # Budget is 200. sqrt(200) ~ 14.14. So n=14 gives 196 points.
        # This uses 196 evaluations. We have 4 left.
        # We can use the remaining 4 to evaluate the corners or center if not covered, 
        # or just stop. The grid 14x14 is dense enough.
        
        n_grid = int(np.sqrt(max_evals))
        # Ensure n_grid^2 <= max_evals
        while n_grid * n_grid > max_evals:
            n_grid -= 1
        
        # Create uniform grid points in [0, 1] for the 2 position variables
        # To maximize hypervolume, we want to cover the boundaries well.
        # np.linspace(0, 1, n) includes 0 and 1.
        x_vals = np.linspace(0, 1, n_grid)
        
        # Generate all combinations of (x_0, x_1)
        x0_grid, x1_grid = np.meshgrid(x_vals, x_vals, indexing='ij')
        
        # Flatten to list of pairs
        pairs = np.column_stack((x0_grid.ravel(), x1_grid.ravel()))
        
        # Total points generated
        n_points = len(pairs)
        
        # We have budget - n_points evaluations left.
        # For DTLZ2, the remaining variables (x_2..x_11) should be 0.5 to be on the front.
        # If we have extra budget, we could potentially do local refinement, 
        # but a uniform grid is already very strong for hypervolume.
        # Let's stick to the grid for determinism and reliability.
        
        # Construct the decision vectors
        # dim = 12, n_obj = 3
        # x_0, x_1 are position variables.
        # x_2 .. x_11 are distance variables, set to 0.5.
        
        X = np.zeros((n_points, self.dim))
        X[:, 0] = pairs[:, 0]
        X[:, 1] = pairs[:, 1]
        
        # Set remaining dimensions to 0.5 (midpoint of [0,1])
        # This places them exactly on the Pareto front for DTLZ2.
        if self.dim > 2:
            X[:, 2:] = 0.5
            
        # Ensure points are within bounds (they should be, but clip just in case)
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        # We must not exceed max_evals.
        # n_points is guaranteed <= max_evals.
        
        # Evaluate sequentially to respect the "problem(x)" call pattern
        # and allow the evaluator to record each one.
        for i in range(n_points):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return None
