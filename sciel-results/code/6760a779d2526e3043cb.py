"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start CMA-ES with adaptive restarts for BBOB f7 (Step ellipsoidal).
    Uses multiple independent runs with different initializations to handle
    the non-convex nature of the step ellipsoidal function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        n = self.dim
        total_budget = self.budget

        best_x = None
        best_fx = np.inf

        # We'll do multiple restarts, each with a portion of the budget
        # For step ellipsoidal, the function has a "step" structure:
        # f(x) = 0.5 * ||x||^2 + ||x|| + sum_{i: x_i > 0} |x_i|
        # The minimum is at the origin, but there are plateaus and steps.
        
        # Strategy: 
        # 1. Start with a few random initializations
        # 2. Run a local CMA-ES-like search from each
        # 3. Keep the best result
        
        rng = np.random.default_rng(123)
        
        # Divide budget among several restarts
        # Use 5 restarts with decreasing budgets
        n_restarts = 5
        budgets = []
        remaining = total_budget
        for i in range(n_restarts):
            if i < n_restarts - 1:
                b = max(20, remaining // 3)
                budgets.append(b)
                remaining -= b
            else:
                budgets.append(remaining)
        
        # Ensure we don't exceed total budget
        while sum(budgets) > total_budget:
            budgets[-1] -= 1
        while sum(budgets) < total_budget:
            budgets[-1] += 1
        
        for restart_idx, b in enumerate(budgets):
            if time.time() > deadline:
                break
            if b < 10:
                continue
            
            # Different initialization strategies for each restart
            if restart_idx == 0:
                # Start near origin (likely good for step ellipsoidal)
                x = rng.uniform(-1, 1, n)
            elif restart_idx == 1:
                # Start at origin
                x = np.zeros(n)
            elif restart_idx == 2:
                # Random in [-2, 2]
                x = rng.uniform(-2, 2, n)
            elif restart_idx == 3:
                # Random in [-5, 5]
                x = rng.uniform(-5, 5, n)
            else:
                # Scaled random
                x = rng.uniform(-3, 3, n)
            
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            
            # Run a (1+1)-ES with adaptive sigma from this point
            sigma = 1.0
            if restart_idx == 0:
                sigma = 0.5
            elif restart_idx == 1:
                sigma = 0.3
            elif restart_idx == 2:
                sigma = 1.0
            elif restart_idx == 3:
                sigma = 2.0
            else:
                sigma = 1.5
            
            # CMA-ES-like adaptation with covariance
            # For simplicity, use diagonal adaptation
            sigma_min = 0.01
            sigma_max = 5.0
            
            # Success counter for 1/5 rule
            successes = 0
            window = 10
            
            for _ in range(b - 1):
                if time.time() > deadline:
                    break
                
                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    # Increase sigma
                    sigma *= 1.3
                    sigma = min(sigma, sigma_max)
                else:
                    # Decrease sigma
                    sigma *= 0.85
                    sigma = max(sigma, sigma_min)
                
                # Reset success window
                if successes >= window:
                    successes = 0
                    sigma *= 1.5
                    sigma = min(sigma, sigma_max)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
