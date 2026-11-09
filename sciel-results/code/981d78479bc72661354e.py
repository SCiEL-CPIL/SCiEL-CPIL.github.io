"""Optimized program for BBOB f15 (Rastrigin rotated) with multi-restart (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive sigma for Rastrigin-type functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Multi-restart strategy:
        # - First restart: 300 calls (60% of budget)
        # - Subsequent restarts: 200 calls each (40% each)
        # This gives us 3 restarts total: 300 + 200 + 200 = 700, but we cap at budget
        # Actually let's do: 250 + 125 + 125 = 500 exactly
        # Or better: first run gets more, subsequent get less
        # Let's do: 280, 110, 110 = 500
        
        restart_budgets = [280, 110, 110]
        
        for restart_idx, rb in enumerate(restart_budgets):
            if total_calls >= budget:
                break
            
            if time.time() > deadline:
                break
            
            # Initialize x for this restart
            # Use diverse initialization: first restart near origin, others random
            if restart_idx == 0:
                # Start near origin with some randomness
                x = rng.uniform(-2, 2, dim)
            else:
                # Random initialization in the box
                x = rng.uniform(-5, 5, dim)
            
            fx = func(x)
            total_calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # (1+1)-ES parameters
            sigma = 2.0
            # For Rastrigin, we want to explore broadly initially
            # then converge
            
            remaining = min(rb - 1, budget - total_calls)
            
            for _ in range(remaining):
                if total_calls >= budget:
                    break
                
                if time.time() > deadline:
                    break
                
                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    # Increase sigma on success
                    sigma *= 1.2
                else:
                    # Decrease sigma on failure
                    sigma *= 0.85
                
                # Track best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                # Keep sigma within reasonable bounds
                sigma = np.clip(sigma, 0.01, 5.0)
            
            # Check time
            if time.time() > deadline:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
