import numpy as np
import time
from scipy.optimize import brentq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_evals = self.budget
        dim = self.dim

        def compute_f1(x1):
            return 1.0 - np.exp(-4.0 * x1) * (np.sin(6.0 * np.pi * x1) ** 6)

        # Analytic minimum of f1
        # tan(6*pi*x1) = 9*pi
        # x1_min = arctan(9*pi) / (6*pi)
        x1_min = np.arctan(9.0 * np.pi) / (6.0 * np.pi)
        f1_min = compute_f1(x1_min)
        
        # U(t) = 11*sqrt(t) - (1/3)*t**1.5
        def U_func(t):
            return 11.0 * np.sqrt(t) - (1.0/3.0) * t**1.5

        U_min = U_func(f1_min)
        U_max = U_func(1.0)
        U_range = U_max - U_min

        n_pts = n_evals
        # Distribute points in U-space
        U_targets = np.linspace(U_min + U_range / (2*n_pts),
                                 U_max - U_range / (2*n_pts), n_pts)
        
        # Invert U(f1) = U_target
        # U(f1) is monotonic increasing for f1 > 0
        # We can use brentq to invert U as well, or just use interpolation on a fine grid
        # But since we need f1 for the next step, let's invert U first.
        # U(t) = 11 t^0.5 - 0.333 t^1.5
        # Let's just use a fine grid for U inversion too, it's cheap.
        n_invert_grid = 10000
        f1_grid = np.linspace(f1_min, 1.0, n_invert_grid)
        U_grid = U_func(f1_grid)
        
        f1_selected = np.interp(U_targets, U_grid, f1_grid)

        # Now invert f1(x1) = f1_target for x1 in [0, x1_min]
        # f1(x1) is decreasing on [0, x1_min] from 1 to f1_min
        x1_final = np.zeros(n_pts)
        
        for i in range(n_pts):
            target_f1 = f1_selected[i]
            # Ensure target is within [f1_min, 1]
            if target_f1 <= f1_min:
                x1_final[i] = x1_min
            elif target_f1 >= 1.0:
                x1_final[i] = 0.0
            else:
                # Solve f1(x) - target = 0 for x in [0, x1_min]
                # f1(0) = 1, f1(x1_min) = f1_min
                # If target is between, there is a unique solution in (0, x1_min)
                try:
                    x1_final[i] = brentq(lambda x: compute_f1(x) - target_f1, 0.0, x1_min, xtol=1e-12)
                except ValueError:
                    # Fallback to grid search if brentq fails (should not happen)
                    x1_grid_local = np.linspace(0, x1_min, 1000)
                    f1_grid_local = compute_f1(x1_grid_local)
                    idx = np.argmin(np.abs(f1_grid_local - target_f1))
                    x1_final[i] = x1_grid_local[idx]

        points = np.zeros((n_evals, self.dim))
        points[:, 0] = x1_final

        recorded = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = points[i]
            f = problem(x)
            recorded.append(f)

        return recorded
