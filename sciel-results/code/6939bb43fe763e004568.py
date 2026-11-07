"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """(1+1)-ES with periodic Nelder-Mead polishing for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget

        # Deadline guard
        start_time = time.time()
        deadline = start_time + 315.0  # Safety margin for 330s limit

        rng = np.random.default_rng(0)

        # Counters
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return float('inf')
            if calls >= total_budget:
                return float('inf')
            
            x = np.clip(x, -5, 5)
            calls += 1
            try:
                fx = func(x)
            except Exception:
                fx = float('inf')
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            return fx

        # Initial point: Center is a good guess for shifted/rotated Weierstrass
        x = np.zeros(dim)
        fx = safe_call(x)
        
        # Step size
        sigma = 1.5
        # 1/5 rule parameters (adapted for faster convergence)
        alpha_up = 1.15
        alpha_down = 1.0 / 1.15
        window = 10
        successes = 0
        
        # Polish interval
        polish_interval = 40
        last_polish_call = calls

        while calls < total_budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            # Clip to box
            y = np.clip(y, -5, 5)

            fy = safe_call(y)
            
            # If budget is exhausted, break immediately
            if calls >= total_budget:
                break

            if fy < fx:
                # Success
                x = y
                fx = fy
                successes += 1
            else:
                # Failure
                pass

            # Update step size every 'window' steps
            if calls % window == 0:
                if successes >= window // 5:
                    sigma *= alpha_up
                else:
                    sigma *= alpha_down
                successes = 0

            # Periodic Polish
            if calls - last_polish_call >= polish_interval and time.time() < deadline - 10:
                last_polish_call = calls
                
                # Use Nelder-Mead for local refinement
                # Start from current best
                x0 = best_x.copy()
                
                def np_func(z):
                    return safe_call(z)
                
                try:
                    res = minimize(np_func, x0, method='Nelder-Mead', 
                                   options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                        # Update current position to the polished best to continue from there
                        x = best_x.copy()
                        fx = best_f
                        # Reset sigma to a moderate value to explore around the new local minimum
                        sigma = 0.5 
                except Exception:
                    pass
                
                # Check if we are out of budget after polish
                if calls >= total_budget:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
