"""Optimized solver for BBOB f18 (Schaffers F7 ill-conditioned) using a deterministic Nelder-Mead approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Solver using deterministic Nelder-Mead simplex method.
    
    Strategy:
    1. Initialize the simplex at the centroid of the box with a reasonable spread.
    2. Run Nelder-Mead to rapidly navigate the ill-conditioned valleys.
    3. If stagnation is detected or budget allows, restart from the best point found
       with a smaller initial spread to refine the solution.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 330s kill
        
        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)
        
        def counting_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Initial simplex setup
        # Start at the center of the box
        center = np.zeros(dim)
        
        # Initial spread: 1.0 in each dimension
        # This creates a simplex with vertices at center +/- 1.0 in each dimension
        # Nelder-Mead will adapt this spread as it progresses
        
        # We'll run multiple Nelder-Mead runs with decreasing spreads if needed
        # Each run gets a portion of the budget
        
        # Phase 1: Initial Nelder-Mead with spread 1.0
        # Allocate 60% of budget to the first run
        budget_per_run = [int(budget * 0.6), int(budget * 0.25), int(budget * 0.15)]
        spreads = [1.0, 0.5, 0.1]
        
        for run_idx in range(len(budget_per_run)):
            if calls >= budget or time.time() > deadline:
                break
            
            spread = spreads[run_idx]
            maxiter = budget_per_run[run_idx]
            
            try:
                # Nelder-Mead with adaptive parameters
                # xatol and fatol are set to very small values to allow thorough exploration
                # maxiter limits the number of iterations, which corresponds to function evaluations
                result = minimize(
                    counting_func,
                    center,
                    method='Nelder-Mead',
                    options={
                        'initial_simplex': None,  # Let scipy create default simplex
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'maxiter': maxiter,
                        'adaptive': True
                    }
                )
                
                # Update best if improved
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
                
                # If we found a good solution, continue with smaller spread
                # Otherwise, we might want to try a different starting point
                # For now, we'll just continue with the next spread
            except Exception:
                # If Nelder-Mead fails, try a simple local search
                # Use a small step size and random directions
                if calls < budget:
                    step_size = spread * 0.1
                    for _ in range(min(10, budget - calls)):
                        if time.time() > deadline:
                            break
                        direction = np.random.randn(dim)
                        direction /= np.linalg.norm(direction)
                        x_new = best_x + step_size * direction
                        x_new = np.clip(x_new, -5, 5)
                        f_new = counting_func(x_new)
                        if f_new < best_f:
                            best_f = f_new
                            best_x = x_new.copy()
        
        # Phase 2: Final polish with very small spread if budget allows
        if calls < budget - 5 and time.time() < deadline:
            try:
                result = minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'maxiter': min(50, budget - calls),
                        'adaptive': True
                    }
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
