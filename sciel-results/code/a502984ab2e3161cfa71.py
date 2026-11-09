"""Rewritten optimizer using a multi-start CMA-ES inspired strategy with adaptive step sizes."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start adaptive optimizer for BBOB-style f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Strategy: Use a combination of global exploration and local exploitation.
        # Since f4 is Büche-Rastrigin, it's multimodal with a global minimum.
        # We'll use a simple evolutionary strategy with multiple restarts or a
        # more sophisticated approach like CMA-ES if budget allows, but given
        # the small budget (500) and dim (5), a well-tuned (1+lambda)-ES or
        # a simple multi-start local search might be effective.
        
        # Let's implement a simple CMA-ES variant or a (1+1)-ES with adaptive sigma
        # and multiple restarts. However, CMA-ES is usually more sample-efficient.
        # With 500 evaluations in 5D, CMA-ES can be quite effective.
        
        # Given the constraints (500 calls), let's implement a simplified CMA-ES
        # or a robust (1+1)-ES with sigma adaptation and restarts.
        
        # Actually, for BBOB functions, CMA-ES is a strong baseline. Let's implement
        # a minimal CMA-ES.
        
        # Parameters for CMA-ES
        # Population size: lambda = 4 + floor(3 * log(dim)) = 4 + 3*log(5) ~ 4+4 = 8
        # But with only 500 evaluations, we can't afford large populations if we want
        # many generations. Let's use a smaller population or a (1+1)-ES.
        
        # Let's try a (1+1)-ES with a more sophisticated sigma adaptation and
        # multiple restarts. We'll divide the budget into several restarts.
        
        # Actually, let's use a simple but effective approach: 
        # 1. Start with a random point.
        # 2. Use (1+1)-ES with sigma adaptation based on success rate.
        # 3. If no improvement for a certain number of steps, restart with a new random point.
        
        best_x = None
        best_f = np.inf
        
        # We'll use a fixed number of restarts, e.g., 5 restarts with 100 evaluations each.
        num_restarts = 5
        evals_per_restart = budget // num_restarts
        
        for restart in range(num_restarts):
            # Initialize
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            
            # If this is the best so far, update
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            sigma = 1.0  # Initial step size
            success_rate = 0.5  # Target success rate for sigma adaptation
            c_sigma = 0.01  # Learning rate for sigma adaptation
            h_sigma = 0.85  # Threshold for sigma adaptation
            
            for i in range(evals_per_restart - 1):
                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                # Update best
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    
                if fy < fx:
                    # Success
                    x, fx = y, fy
                    success_rate = (1 - c_sigma) * success_rate + c_sigma
                else:
                    # Failure
                    success_rate = (1 - c_sigma) * success_rate
                    
                # Adapt sigma based on success rate
                # If success rate is high, increase sigma; if low, decrease
                if success_rate > h_sigma:
                    sigma *= 1.05
                elif success_rate < (1 - h_sigma):
                    sigma *= 0.95
                else:
                    sigma *= 1.0  # No change
                
                # Ensure sigma stays within reasonable bounds
                sigma = np.clip(sigma, 0.01, 5.0)
                
                # Early stop if we've converged (very small steps and no improvement)
                if sigma < 0.01 and i > 10:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
