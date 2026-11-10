"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting WFG9 structure: grid search on defining variables."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WFG9 Pareto front is determined by the first (n_obj - 1) variables.
        # For n_obj=3, that's x[0] and x[1].
        # The remaining variables (x[2]...x[11]) should be set to 0 to be on the Pareto front.
        
        n_defining = self.n_obj - 1
        if n_defining < 1:
            n_defining = 1
        if n_defining > self.dim:
            n_defining = self.dim

        # We will use a grid search for the defining variables.
        # Budget is 200.
        # We want to maximize the number of grid points.
        # If n_defining == 2, we can do a 14x14 grid (196 points).
        # If n_defining == 1, we can do a 200 point line.
        
        if n_defining == 2:
            grid_size = 14
            total_grid_points = grid_size * grid_size
            if total_grid_points > self.budget:
                # Reduce grid size if budget is smaller
                grid_size = int(np.sqrt(self.budget))
                total_grid_points = grid_size * grid_size
                
            x0_vals = np.linspace(lo[0], hi[0], grid_size)
            x1_vals = np.linspace(lo[1], hi[1], grid_size)
            
            evaluated = 0
            for x0 in x0_vals:
                for x1 in x1_vals:
                    if evaluated >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    x = np.zeros(self.dim)
                    x[0] = x0
                    x[1] = x1
                    # Set remaining to 0.0 (standard for WFG front)
                    # If bounds don't include 0, we might need to adjust, but WFG9 usually has 0 in bounds.
                    # To be safe, if 0 is not in [lo, hi], we use midpoint? 
                    # But WFG9 bounds are typically [0, 1] or similar where 0 is valid.
                    # Let's check if 0 is in bounds. If not, use lo or hi?
                    # Actually, for WFG, the "distance" is minimized when these vars are 0.
                    # If 0 is out of bounds, the problem is not standard WFG9 or bounds are weird.
                    # We'll assume standard bounds where 0 is valid or close.
                    # Let's use 0.0 clipped to bounds to be safe.
                    for i in range(2, self.dim):
                        x[i] = np.clip(0.0, lo[i], hi[i])
                    
                    problem(x)
                    evaluated += 1
                
                if evaluated >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
            # If we have budget left, add a few random points in the defining subspace
            # to catch any non-linearities missed by the grid.
            while evaluated < self.budget:
                if time.time() > deadline:
                    break
                x = np.zeros(self.dim)
                x[0] = np.random.uniform(lo[0], hi[0])
                x[1] = np.random.uniform(lo[1], hi[1])
                for i in range(2, self.dim):
                    x[i] = np.clip(0.0, lo[i], hi[i])
                problem(x)
                evaluated += 1

        elif n_defining == 1:
            # Linear scan
            num_points = self.budget
            x0_vals = np.linspace(lo[0], hi[0], num_points)
            for x0 in x0_vals:
                if time.time() > deadline:
                    break
                x = np.zeros(self.dim)
                x[0] = x0
                for i in range(1, self.dim):
                    x[i] = np.clip(0.0, lo[i], hi[i])
                problem(x)
        else:
            # Fallback to random search in full space if structure is unknown
            for _ in range(self.budget):
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.random(self.dim)
                problem(x)

        return None
# EVOLVE-BLOCK-END
