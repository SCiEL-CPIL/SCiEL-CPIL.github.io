import numpy as np
import time
import itertools
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        calls_made = 0
        best_x = None
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget or time.time() > deadline - 1:
                raise RuntimeError("Budget or time exceeded")
            calls_made += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Coarse Grid Search (3^5 = 243 calls)
        try:
            points_1d = np.linspace(-5.0, 5.0, 3)
            grid_1 = list(itertools.product(*[points_1d] * self.dim))
            
            for x in grid_1:
                x_arr = np.array(x)
                counted_func(x_arr)
            
            # Determine the refinement cell based on best_x
            midpoints = (points_1d[:-1] + points_1d[1:]) / 2.0
            indices = []
            for d in range(self.dim):
                val = best_x[d]
                if val < midpoints[0]:
                    indices.append(0)
                elif val < midpoints[1]:
                    indices.append(1)
                else:
                    indices.append(2)
            
            bounds_low = []
            bounds_high = []
            for d in range(self.dim):
                idx = indices[d]
                if idx == 0:
                    bounds_low.append(-5.0)
                    bounds_high.append(midpoints[0])
                elif idx == 1:
                    bounds_low.append(midpoints[0])
                    bounds_high.append(midpoints[1])
                else:
                    bounds_low.append(midpoints[1])
                    bounds_high.append(5.0)
            
            # Phase 2: Fine Grid Search in the best cell (3^5 = 243 calls)
            points_1d_fine = []
            for d in range(self.dim):
                lo = bounds_low[d]
                hi = bounds_high[d]
                points_1d_fine.append(np.linspace(lo, hi, 3))
                
            grid_2 = list(itertools.product(*points_1d_fine))
            
            for x in grid_2:
                x_arr = np.array(x)
                counted_func(x_arr)

        except Exception:
            pass

        # Phase 3: Local Polish with Nelder-Mead
        if best_x is not None and calls_made < self.budget and time.time() < deadline - 5:
            remaining = self.budget - calls_made
            maxiter_nm = max(1, remaining // (self.dim + 1))
            
            try:
                minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': maxiter_nm, 'xatol': 1e-8, 'fatol': 1e-10}
                )
            except Exception:
                pass

        # Final safety check: if no valid best_x, do one call
        if best_x is None:
            try:
                x0 = np.zeros(self.dim)
                counted_func(x0)
            except Exception:
                best_x = np.zeros(self.dim)
                best_f = 0.0

        return best_x, best_f
