import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        total_budget = self.budget
        
        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > total_budget:
                raise ValueError("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        bounds = [(-5.0, 5.0)] * n

        # Phase 1: Coordinate descent warm start for early progress
        # Try several random starting points, do coordinate descent on each
        cd_budget = int(total_budget * 0.4)
        n_starts = 5
        
        for s in range(n_starts):
            if time.time() > deadline or calls_made[0] >= total_budget:
                break
            
            # Random start within bounds
            x = np.random.uniform(-4.0, 4.0, n)
            fx = wrapped_func(x)
            
            # Coordinate descent: sweep through dimensions, try small perturbations
            step_sizes = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]
            for step in step_sizes:
                if time.time() > deadline or calls_made[0] >= total_budget:
                    break
                improved = True
                while improved:
                    improved = False
                    for d in range(n):
                        if time.time() > deadline or calls_made[0] >= total_budget:
                            break
                        # Try positive and negative steps
                        for direction in [1.0, -1.0]:
                            x_trial = x.copy()
                            x_trial[d] += direction * step
                            x_trial = np.clip(x_trial, -5.0, 5.0)
                            f_trial = wrapped_func(x_trial)
                            if f_trial < fx:
                                x = x_trial
                                fx = f_trial
                                improved = True
                                break  # move to next dimension after improvement
        
        # Phase 2: Nelder-Mead refinement from best point
        if time.time() < deadline - 5.0 and calls_made[0] < total_budget - 10:
            try:
                res_nm = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12,
                             'initial_simplex': None}
                )
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = np.clip(res_nm.x, -5.0, 5.0)
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        # Phase 3: Additional Nelder-Mead passes if budget remains
        remaining = total_budget - calls_made[0]
        if remaining > 20 and time.time() < deadline - 5.0:
            try:
                res_nm2 = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': min(remaining // 10, 300), 'xatol': 1e-14, 'fatol': 1e-14}
                )
                if res_nm2.fun < best_f:
                    best_f = res_nm2.fun
                    best_x = np.clip(res_nm2.x, -5.0, 5.0)
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        return best_x, best_f
