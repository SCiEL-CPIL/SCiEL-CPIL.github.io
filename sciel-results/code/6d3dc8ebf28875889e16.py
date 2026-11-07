"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Sobol sequence exploration followed by (1+1)-ES refinement for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        total_calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, total_calls
            x = np.clip(x, -5, 5)
            f = func(x)
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Sobol sequence exploration
        # Use a Sobol sequence to systematically cover the space
        # We'll allocate about 30% of budget to exploration
        explore_budget = min(150, budget // 3)
        
        # Generate Sobol sequence points in [0, 1]^n
        sampler = qmc.Sobol(d=n, scramble=True)
        # We need enough points; Sobol requires power of 2
        m = max(1, int(np.log2(explore_budget)))
        num_points = 2 ** m
        if num_points > explore_budget:
            num_points = explore_budget
            m = int(np.log2(num_points))
        
        # Generate the Sobol points
        sobol_points = sampler.random(num_points)  # in [0, 1]^n
        # Map to [-5, 5]^n
        sobol_points = sobol_points * 10.0 - 5.0
        
        # Evaluate Sobol points
        for i in range(num_points):
            if time.time() > deadline or total_calls >= budget:
                break
            x = sobol_points[i]
            evaluate(x)
        
        # Phase 2: (1+1)-ES refinement from the best point found
        if best_x is None:
            # Fallback if no points were evaluated
            x = np.zeros(n)
            evaluate(x)
        
        x = best_x.copy()
        f_x = best_f
        sigma = 1.0  # Start with moderate step size
        
        remaining_calls = budget - total_calls
        last_improve_call = total_calls
        
        while remaining_calls > 0:
            if time.time() > deadline:
                break
            
            # Check for stagnation: if sigma is very small and no improvement recently
            if sigma < 1e-4 and (total_calls - last_improve_call) > 50:
                # Try to escape: kick from best with larger sigma
                x = best_x + np.random.standard_normal(n) * 1.0
                x = np.clip(x, -5, 5)
                f_x = evaluate(x)
                remaining_calls -= 1
                if f_x < best_f:
                    last_improve_call = total_calls
                else:
                    # Reset to best and try with larger sigma
                    x = best_x.copy()
                    f_x = best_f
                    sigma = 0.5
                continue
            
            # Generate offspring
            z = np.random.standard_normal(n)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            
            f_y = evaluate(y)
            remaining_calls -= 1
            
            if f_y < f_x:
                x = y
                f_x = f_y
                sigma *= 1.05
                last_improve_call = total_calls
            else:
                sigma *= 0.95
            
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)
            
            # If we have very few calls left, reduce sigma to ensure we don't jump out of the basin
            if remaining_calls < 20:
                sigma = min(sigma, 0.01)
        
        return best_x, best_f

# EVOLVE-BLOCK-END
