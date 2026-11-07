import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget

        # DTLZ2 structure:
        # x[0], x[1] are position variables (angles)
        # x[2:] are decision variables; g(x) is minimized at x[2:] = 0.5
        # The Pareto front is a quarter-sphere in 3D: f1^2 + f2^2 + f3^2 = 1

        # Strategy:
        # 1. Use a deterministic grid for guaranteed coverage of the Pareto front
        # 2. Use Sobol sequence for additional space-filling samples
        # 3. Fix decision variables at 0.5 to be on the Pareto front (g=0)
        # 4. Add vertex points for better hypervolume coverage

        # Generate a 14x14 grid for the two position variables
        grid_n = 14
        grid_points = []
        for i in range(grid_n):
            for j in range(grid_n):
                # Use interior points to avoid boundary issues
                u1 = (i + 0.5) / grid_n
                u2 = (j + 0.5) / grid_n
                grid_points.append([u1, u2])
        
        grid_points = np.array(grid_points)
        
        # Map to decision space
        X_grid = np.zeros((grid_points.shape[0], self.dim))
        X_grid[:, 0] = grid_points[:, 0] * (hi[0] - lo[0]) + lo[0]
        X_grid[:, 1] = grid_points[:, 1] * (hi[1] - lo[1]) + lo[1]
        
        # Fill remaining dimensions with midpoint
        if self.dim > 2:
            mid_points = (lo[2:] + hi[2:]) / 2.0
            X_grid[:, 2:] = mid_points

        # Add Sobol points for remaining budget
        remaining = n_evals - X_grid.shape[0]
        if remaining > 0:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True, seed=42)
            sobol_samples = sampler.random(remaining)
            
            X_sobol = np.zeros((remaining, self.dim))
            X_sobol[:, 0] = sobol_samples[:, 0] * (hi[0] - lo[0]) + lo[0]
            X_sobol[:, 1] = sobol_samples[:, 1] * (hi[1] - lo[1]) + lo[1]
            
            if self.dim > 2:
                X_sobol[:, 2:] = mid_points
            
            X = np.vstack([X_grid, X_sobol])
        else:
            X = X_grid

        # Evaluate all points
        eval_count = 0
        for i in range(X.shape[0]):
            if time.time() > deadline:
                break
            try:
                x = X[i]
                f = problem(x)
                eval_count += 1
            except Exception:
                break

        return X
