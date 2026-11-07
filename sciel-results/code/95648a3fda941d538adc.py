import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 300.0  # 300s limit, leave margin for safety

        def time_left():
            return deadline - time.time()

        # (1+1)-ES with 1/5 success rule
        # This is a simple, robust optimizer for low-dimensional problems
        # Schaffers F7 is multimodal but (1+1)-ES with proper step size adaptation works well

        # Initial point: try origin first, then other heuristics if needed
        x = np.zeros(dim)
        f = func(x)
        calls = 1
        best_x = x.copy()
        best_f = f

        # Step size
        sigma = 1.0
        # 1/5 rule parameters
        success_count = 0
        step_count = 0
        sigma_factor_up = 1.05
        sigma_factor_down = 1.0 / 1.05
        target_success_rate = 0.2

        # For (1+1)-ES, we generate one candidate per step
        # But to be more efficient, we can do multiple steps per "iteration"
        # Let's do a simple (1+1)-ES loop

        # Add some diversity: try a few random starting points if budget allows
        # But keep it simple: start at origin

        while calls < budget and time_left() > 1.0:
            # Generate candidate
            z = np.random.standard_normal(dim)
            x_new = x + sigma * z

            # Evaluate
            f_new = func(x_new)
            calls += 1

            # Update best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

            # 1/5 success rule
            if f_new < f:
                success_count += 1
                x = x_new
                f = f_new
            else:
                # Keep x unchanged
                pass

            step_count += 1

            # Adapt sigma every 10 steps
            if step_count >= 10:
                success_rate = success_count / step_count
                if success_rate > target_success_rate:
                    sigma *= sigma_factor_up
                else:
                    sigma *= sigma_factor_down

                # Reset counters
                success_count = 0
                step_count = 0

                # Clamp sigma to reasonable range
                sigma = max(0.01, min(sigma, 10.0))

        # Final high-precision polish pass using Nelder-Mead
        if time_left() > 5.0 and calls < budget - 100:
            # Try to polish the best solution found
            try:
                result = minimize(func, best_x, method='Nelder-Mead', 
                                  options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except:
                pass

        return best_x, best_f
