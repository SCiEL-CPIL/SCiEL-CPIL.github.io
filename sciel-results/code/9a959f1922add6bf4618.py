"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Robust optimizer using Random Search and (1+1)-ES for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s kill
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0
        max_calls = self.budget - 5  # Buffer to avoid edge cases

        def safe_func(x):
            nonlocal calls_used, best_f, best_x
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            x = np.clip(x, -5.0, 5.0)
            if calls_used >= max_calls:
                return best_f if best_f != np.inf else 0.0
            calls_used += 1
            f_val = float(func(x))
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def check_deadline():
            return time.time() > deadline or calls_used >= max_calls

        # Phase 1: Random Search for Global Exploration
        # Use ~20% of budget for initial random sampling to find a good basin
        n_random = max(10, int(0.2 * max_calls))
        for i in range(n_random):
            if check_deadline():
                break
            x = rng.uniform(-5, 5, self.dim)
            safe_func(x)

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = safe_func(best_x)

        # Phase 2: (1+1)-ES for Local Refinement
        # Start from the best point found
        x_current = best_x.copy()
        sigma = 1.0  # Initial step size
        no_improve = 0
        max_no_improve = 20  # Restart if no improvement for 20 steps

        while not check_deadline():
            # Generate one offspring
            y = rng.standard_normal(self.dim)
            x_new = x_current + sigma * y
            x_new = np.clip(x_new, -5, 5)
            
            f_new = safe_func(x_new)
            
            if f_new < best_f:
                # Accept
                x_current = x_new
                no_improve = 0
                # Increase step size slightly
                sigma *= 1.1
            else:
                no_improve += 1
                # Decrease step size
                sigma *= 0.85
            
            # Check for stagnation and restart locally if stuck
            if no_improve >= max_no_improve:
                # Try a few random restarts around the current best or global best
                for _ in range(5):
                    if check_deadline():
                        break
                    x_restart = best_x + rng.uniform(-1, 1, self.dim)
                    x_restart = np.clip(x_restart, -5, 5)
                    safe_func(x_restart)
                
                # Reset local search state
                x_current = best_x.copy()
                sigma = 1.0
                no_improve = 0
                
                # If we've used up most of the budget, break
                if calls_used > 0.8 * max_calls:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
