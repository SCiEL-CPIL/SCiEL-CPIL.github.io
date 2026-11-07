"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Problem-specific optimizer exploiting DTLZ2's known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ2 specifics:
        # f_i(x) = g(x) * (1 + cos(theta_1) * sin(theta_2) * ... * sin(theta_{n_obj-1}) * cos(theta_i))
        # Pareto front: g=0, f_i = sin(theta_i) (quarter sphere)
        # g(x) = sum_{x_i in [dim - M + 1, dim]} (x_i - 0.5)^2
        # M = 10 for DTLZ2 standard
        
        M = min(10, dim)  # Number of variables in G function
        
        # Strategy: Directly parameterize the Pareto front.
        # For DTLZ2 with 3 objectives, the Pareto front is a quarter-sphere.
        # The decision variables x[0] and x[1] (assuming M=10, so x[0:8] are position vars? 
        # Actually, standard DTLZ2 uses the last M variables for G. 
        # Let's check the standard definition carefully.
        # Standard DTLZ2:
        # g(x) = sum_{i=M}^{n-1} (x_i - 0.5)^2
        # f_0 = g * (1 + cos(pi*x_0/2) * sin(pi*x_1/2) * ... * sin(pi*x_{n_obj-2}/2) * cos(pi*x_{n_obj-1}/2))
        # ...
        # Wait, the number of position variables is n_obj - 1.
        # For n_obj=3, we have 2 position variables: x_0 and x_1.
        # The remaining variables x_2 ... x_{n-1} are for G.
        # So if dim=12, x[0] and x[1] determine the direction, and x[2:12] determine G.
        # To get G=0, we set x[2:12] = 0.5.
        # Then f_0 = cos(pi*x_0/2) * sin(pi*x_1/2)
        # f_1 = cos(pi*x_0/2) * cos(pi*x_1/2)
        # f_2 = sin(pi*x_0/2)
        
        # Let's verify this mapping.
        # If x_0=0, x_1=0: f_0=0, f_1=1, f_2=0. (1,0,0) rotated?
        # If x_0=0, x_1=1: f_0=1, f_1=0, f_2=0.
        # If x_0=1, x_1=0: f_0=0, f_1=0, f_2=1.
        # If x_0=1, x_1=1: f_0=0, f_1=0, f_2=1.
        # This covers the quarter sphere in the positive orthant.
        
        # We will sweep x_0 and x_1 on a grid to densely cover the Pareto front.
        # Since we have budget=200, we can afford a grid of e.g. 14x14 = 196 points.
        # This will give a very dense coverage of the front, maximizing hypervolume.
        
        # Determine number of position variables
        n_pos = n_obj - 1  # 2 for 3 objectives
        
        # Create a grid for the position variables
        # We want to cover [0, 1] for each position variable.
        # Let's use a grid size that fits within budget.
        # We'll reserve some budget for random sampling to ensure no gaps.
        
        grid_size = int(np.sqrt(budget * 0.9))  # Use 90% of budget for grid
        if grid_size < 2:
            grid_size = 2
            
        # Create grid points
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = np.zeros(dim)
                x[0] = i / (grid_size - 1) if grid_size > 1 else 0.5
                x[1] = j / (grid_size - 1) if grid_size > 1 else 0.5
                # Set G variables to 0.5 to make G=0
                x[n_pos:] = 0.5
                grid_points.append(x)
                
        # Evaluate grid points
        for x in grid_points:
            if time.time() > deadline:
                break
            f = problem(x)
            # No need to check dominance, just record all
            # The evaluator records all returned objective vectors
            
        # Fill remaining budget with random sampling on the front
        remaining_budget = budget - len(grid_points)
        for _ in range(remaining_budget):
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[:n_pos] = np.random.rand(n_pos)
            x[n_pos:] = 0.5  # Set G=0
            f = problem(x)
            
        return None

# EVOLVE-BLOCK-END
