"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES-like optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        best_x = None
        best_f = np.inf
        calls_used = 0
        remaining = self.budget

        # Use multiple restarts with CMA-ES-like strategy
        # For a 5D problem with 500 budget, we can afford a few restarts
        
        num_restarts = 3
        budget_per_restart = remaining // num_restarts
        extra = remaining % num_restarts
        
        for restart in range(num_restarts):
            b = budget_per_restart + (1 if restart < extra else 0)
            if b <= 0:
                break
                
            # Initialize
            x = rng.uniform(-4, 4, self.dim)
            fx = func(x)
            calls_used += 1
            remaining -= 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 1.0
            # Population size for CMA-ES-like: lambda = 4 + floor(3*log(dim))
            lam = max(4, 4 + int(3 * np.log(self.dim)))
            # Cap lambda to not exceed remaining budget
            lam = min(lam, max(2, remaining // 2))
            
            # Simple evolution strategy with population
            for gen in range(remaining):
                if remaining <= 0:
                    break
                
                # Generate offspring
                for i in range(lam):
                    if remaining <= 0:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = func(y)
                    remaining -= 1
                    calls_used += 1
                    
                    if fy < fx:
                        # Success: move towards y
                        x = y
                        fx = fy
                        sigma *= 1.1
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        # Failure: shrink sigma
                        sigma *= 0.95
                
                # Adaptive sigma based on success rate (1/5 rule approximation)
                # Simple: if sigma gets too small or too large, adjust
                sigma = np.clip(sigma, 0.01, 5.0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
