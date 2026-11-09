"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and multi-restart strategy."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        
        # Strategy: 8 restarts with decreasing budget allocation
        # First restart gets more budget, later ones get less
        # This allows early exploration and later exploitation
        n_restarts = 8
        # Budget allocation: geometrically decreasing
        # Total = 500 calls
        # First restart: ~150, then decreasing
        budgets = [150, 80, 60, 50, 40, 35, 30, 25]
        # Adjust to fit budget
        total_alloc = sum(budgets)
        scale = budget / total_alloc
        budgets = [max(1, int(b * scale)) for b in budgets]
        # Ensure we don't exceed budget
        while sum(budgets) > budget:
            budgets[-1] -= 1
        while sum(budgets) < budget:
            budgets[0] += 1
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            if budget - sum(budgets[:restart]) <= 0:
                break
            
            # Diverse initial points
            if restart == 0:
                x = np.zeros(dim)
            elif restart == 1:
                x = rng.uniform(-4, 4, dim)
            elif restart == 2:
                x = rng.uniform(-4, 4, dim)
            elif restart == 3:
                x = rng.uniform(-3.5, 3.5, dim)
            elif restart == 4:
                x = rng.uniform(-3, 3, dim)
            elif restart == 5:
                x = rng.uniform(-2.5, 2.5, dim)
            elif restart == 6:
                # Perturb around best found so far
                if best_x is not None:
                    x = np.clip(best_x + rng.normal(0, 1.5, dim), -5, 5)
                else:
                    x = rng.uniform(-2, 2, dim)
            else:
                # Fine-tune around best
                if best_x is not None:
                    x = np.clip(best_x + rng.normal(0, 0.8, dim), -5, 5)
                else:
                    x = rng.uniform(-2, 2, dim)
            
            fx = func(x)
            
            # Update global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # (1+1)-ES with tuned parameters
            # Start with smaller sigma for finer search
            sigma = 1.0 if restart < 3 else 0.5
            
            b = budgets[restart]
            for _ in range(b - 1):
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3  # Slower growth
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9  # Slower decay
                
                # Keep sigma within reasonable bounds
                if sigma < 1e-5:
                    sigma = 1e-5
                if sigma > 8.0:
                    sigma = 8.0
        
        return best_x, best_f
# EVOLVE-BLOCK-END
