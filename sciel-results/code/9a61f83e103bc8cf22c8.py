import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280  # Safety margin before 300s limit

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Wrap func to count calls and check deadline
        def wrapped_func(x):
            if time.time() > self.deadline:
                raise TimeoutError()
            if self.calls >= budget:
                raise ValueError("Budget exceeded")
            self.calls += 1
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            val = func(x)
            if np.isnan(val) or np.isinf(val):
                val = 1e10
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # Phase 1: Coarse grid search to identify basin of attraction
        # Use a grid with 3 points per dimension in [-4, 4]
        # That's 3^5 = 243 evaluations
        grid_points = []
        coords = [-4.0, 0.0, 4.0]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            grid_points.append(np.array([coords[i], coords[j], coords[k], coords[l], coords[m]]))

        # Evaluate all grid points
        grid_best_x = None
        grid_best_f = np.inf
        for gp in grid_points:
            if self.calls >= budget or time.time() > self.deadline:
                break
            f_val = wrapped_func(gp)
            if f_val < grid_best_f:
                grid_best_f = f_val
                grid_best_x = gp.copy()

        # Phase 2: Local refinement using Nelder-Mead from the best grid point
        if grid_best_x is not None and self.calls < budget - 10 and time.time() < self.deadline:
            try:
                res_nm = minimize(
                    wrapped_func,
                    grid_best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        # Phase 3: If we have budget left, try L-BFGS-B from the best point for gradient-based refinement
        if self.calls < budget - 10 and time.time() < self.deadline:
            best_start = self.best_x if self.best_x is not None else np.zeros(dim)
            try:
                res_lbfgs = minimize(
                    wrapped_func,
                    best_start,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'gtol': 1e-8}
                )
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        # Phase 4: Final Nelder-Mead polish if budget allows
        if self.calls < budget - 5 and time.time() < self.deadline:
            best_start = self.best_x if self.best_x is not None else np.zeros(dim)
            try:
                res_nm2 = minimize(
                    wrapped_func,
                    best_start,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        # If no valid result was found, return a safe point
        if self.best_x is None:
            self.best_x = np.zeros(dim)
            self.best_f = wrapped_func(self.best_x)

        return self.best_x, self.best_f
