"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, restarts, and center-biased initialization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim
        
        # Adaptive restart strategy
        max_restarts = 20
        # Start with fewer calls per restart, increase as we go
        base_calls_per_restart = max(1, budget // max_restarts)
        
        # Track stagnation for early restart
        stagnation_threshold = 25
        
        for restart in range(max_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            
            # Adaptive calls per restart: start small, grow
            # First few restarts get fewer calls to explore quickly
            if restart < 3:
                calls_this_restart = max(10, base_calls_per_restart)
            elif restart < 8:
                calls_this_restart = max(20, base_calls_per_restart * 2)
            else:
                calls_this_restart = max(30, base_calls_per_restart * 3)
            
            # Ensure we don't exceed remaining budget
            remaining = budget - calls_made
            calls_this_restart = min(calls_this_restart, remaining)
            
            if calls_this_restart <= 0:
                break
            
            # Initialize with random point, biased toward center for early restarts
            if restart < 4:
                x = rng.uniform(-3, 3, dim)
            elif restart < 8:
                x = rng.uniform(-4, 4, dim)
            else:
                x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            
            # (1+1)-ES with adaptive sigma
            sigma = 1.5
            no_improve = 0
            sigma_min = 0.005
            sigma_max = 15.0
            
            for step in range(calls_this_restart):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Generate trial point
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    # Success: move to y
                    x, fx = y, fy
                    no_improve = 0
                    # Increase sigma moderately on success
                    sigma = min(sigma * 1.15, sigma_max)
                else:
                    # Failure: stay at x
                    no_improve += 1
                    # Decrease sigma on failure
                    sigma = max(sigma * 0.92, sigma_min)
                
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                # Early restart if stagnating
                if no_improve > stagnation_threshold:
                    break
            
            # If we found a very good solution, we can stop early
            if best_f < -80:
                break
            # If we've made significant progress, reduce restarts
            if best_f < -40 and restart >= 5:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
