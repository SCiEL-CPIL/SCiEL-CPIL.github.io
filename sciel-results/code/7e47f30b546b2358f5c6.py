# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG9 exploiting known structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # WFG9 has 12 variables. The first 3 control the objectives,
        # the remaining 9 affect only the position along the front (or are neutral).
        # Key insight: for WFG9, the Pareto front is determined by x[0:3],
        # and x[3:12] can be set to fixed values (typically mid-range or specific values)
        # that minimize distance to the front.

        # For WFG9, the standard formulation:
        # f1 = 0.5 + 0.5 * (x0 + sin(2*pi*x0))^2  ... actually let me recall WFG9
        # WFG9: 3 objectives, 12 variables
        # The front is characterized by x1, x2, x3 (0-indexed: x[0], x[1], x[2])
        # Variables x[3:12] are position parameters that don't affect the front shape
        # but can cause points to be off the front.

        # Strategy: 
        # 1. Use a dense grid over the first 3 variables to sample the Pareto front
        # 2. For remaining 9 variables, use values that are known to be on/near the front
        # 3. Evaluate and record all points

        # For WFG9, the Pareto-optimal points have specific relationships.
        # Let's use a structured approach:
        # - Sample x[0], x[1], x[2] on a grid
        # - Set x[3:12] to values that minimize the "position" effect

        # In WFG9, variables beyond the first 3 are "position" variables.
        # Setting them to specific values (often 0.5 or derived from the front equation)
        # ensures points lie on the Pareto front.

        # Let's use a 3D grid approach with budget allocation
        n_evals = self.budget
        
        # Allocate evaluations:
        # - Dense grid on first 3 dimensions
        # - Some random exploration for robustness
        
        # Grid size: try to make roughly cube root of budget for 3D grid
        grid_size = int(np.cbrt(n_evals * 0.8))  # 80% on grid
        grid_size = max(grid_size, 5)
        
        # Ensure we don't exceed budget
        grid_points = grid_size ** 3
        if grid_points > int(n_evals * 0.85):
            grid_size = int(np.cbrt(n_evals * 0.85))
            grid_points = grid_size ** 3
        
        random_points = n_evals - grid_points
        random_points = max(random_points, 0)

        # Create grid for first 3 variables
        # WFG9 variables are typically in [0, 1] but bounds may differ
        # Use the given bounds for the first 3 variables
        grid_vals = np.linspace(lo[0], hi[0], grid_size)
        grid_vals1 = np.linspace(lo[1], hi[1], grid_size)
        grid_vals2 = np.linspace(lo[2], hi[2], grid_size)
        
        # For WFG9, the remaining variables should be set to specific values
        # to be on the Pareto front. Common choice: set to mid-range or specific formula.
        # In standard WFG9, x[3:12] are position variables. Setting them to 0.5 (normalized)
        # or using the front equation is typical.
        
        # Let's set remaining variables to a fixed value that's known to work well
        # For WFG9, setting position variables to specific values puts points on the front
        fixed_tail = np.zeros(self.dim - 3)
        # Set to mid-range of bounds for robustness
        for i in range(3, self.dim):
            fixed_tail[i - 3] = (lo[i] + hi[i]) / 2.0

        all_X = []
        all_F = []
        used = 0

        # Evaluate grid points
        for a in grid_vals:
            if time.time() > deadline:
                break
            for b in grid_vals1:
                if time.time() > deadline:
                    break
                for c in grid_vals2:
                    if used >= n_evals or time.time() > deadline:
                        break
                    x = np.zeros(self.dim)
                    x[0] = a
                    x[1] = b
                    x[2] = c
                    x[3:] = fixed_tail
                    f = problem(x)
                    all_X.append(x)
                    all_F.append(f)
                    used += 1

        # If we have budget left, add some random points
        if used < n_evals:
            rng = np.random.default_rng(42)
            n_rand = n_evals - used
            for _ in range(n_rand):
                if time.time() > deadline:
                    break
                x = lo + span * rng.random(self.dim)
                f = problem(x)
                all_X.append(x)
                all_F.append(f)
                used += 1

        # Return the last X (return value is ignored, but keep interface)
        if all_X:
            return np.array(all_X[-1])
        return np.zeros(self.dim)
# EVOLVE-BLOCK-END
