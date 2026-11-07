"""Optimizer for BBOB-style f15 (Rastrigin rotated) with Multi-start (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with 1/5 success rule for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0

        # Use 7 restarts to balance exploration and exploitation
        n_restarts = 7
        calls_per_restart = budget // n_restarts

        for r in range(n_restarts):
            # Check time and budget
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Initialize restart point with random uniform distribution
            x = rng.uniform(-5, 5, dim)
            
            # Evaluate initial point
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # (1+1)-ES loop for this restart
            sigma = 2.0  # Initial step size
            
            max_calls_this_restart = calls_per_restart - 1
            if max_calls_this_restart < 0:
                continue

            for _ in range(max_calls_this_restart):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    # Success
                    x = y
                    fx = fy
                    sigma *= 1.5
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Failure
                    sigma *= 0.87

        # Final refinement on the best point found
        if best_x is not None and calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5 # Smaller sigma for refinement
            remaining = budget - calls_made
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2 # Less aggressive growth for refinement
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
