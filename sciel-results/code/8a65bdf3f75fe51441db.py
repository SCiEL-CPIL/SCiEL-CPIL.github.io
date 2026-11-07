"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive step sizes for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        
        # Track calls
        call_count = 0
        
        def safe_func(x):
            nonlocal call_count
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            call_count += 1
            return f
        
        # Strategy: Multi-start with local polishing
        # For Rastrigin, the function is f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # The global minimum is at x=0 (before rotation/shift), value = 0 (before offset)
        # With rotation and shift, we need to explore broadly.
        
        # Phase 1: Broad exploration with multiple restarts
        # Use a population-based approach to escape local minima
        
        # First, do a coarse grid search to identify promising regions
        # Since dim=5, we can do a small grid
        # But 500 calls is limited. Let's use a smarter approach.
        
        # Approach: 
        # 1. Start with several random points
        # 2. For each, do a local (1+1)-ES with adaptive step
        # 3. Keep the best
        # 4. If time allows, do more restarts from different regions
        
        # Given 500 calls, let's allocate:
        # - 50 calls for initial broad sampling (10 points x 5 evals each? No, just 50 random)
        # - Actually, let's do: 10 restarts, each with ~50 calls for local search
        # - But early progress matters for AOCC, so we want to find good values quickly
        
        # Better strategy:
        # - Start with a few points, evaluate them
        # - Use the best as starting point for (1+1)-ES
        # - Periodically restart from random points if stuck
        
        # Let's implement a multi-start (1+1)-ES with:
        # - Adaptive per-dimension step sizes
        # - Restart mechanism when stuck
        
        # Initialize with random points
        n_initial = 10
        initial_points = rng.uniform(-5, 5, (n_initial, dim))
        initial_values = []
        for p in initial_points:
            if call_count >= budget or time.time() > deadline:
                break
            f = safe_func(p)
            initial_values.append((f, p.copy()))
        
        if not initial_values:
            # Couldn't even make initial calls
            return np.zeros(dim), np.inf
        
        # Sort by fitness
        initial_values.sort(key=lambda x: x[0])
        best_f, best_x = initial_values[0]
        
        # Now do local optimization from the best point, with occasional restarts
        x = best_x.copy()
        fx = best_f
        
        # Per-dimension step sizes, start moderately
        sigma = np.ones(dim) * 1.0
        
        # Track stagnation
        no_improve_count = 0
        max_no_improve = 15
        
        # Main loop
        while call_count < budget and time.time() < deadline:
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = safe_func(y)
            
            if fy < fx:
                x, fx = y, fy
                no_improve_count = 0
                # Increase step size on improvement
                sigma *= 1.2
                # Cap step size
                sigma = np.minimum(sigma, 3.0)
            else:
                no_improve_count += 1
                # Decrease step size
                sigma *= 0.85
                # Floor step size
                sigma = np.maximum(sigma, 0.01)
                
                # If stuck, try a restart
                if no_improve_count >= max_no_improve:
                    # Random restart from a different region
                    # But keep the best so far
                    x_restart = rng.uniform(-5, 5, dim)
                    fx_restart = safe_func(x_restart)
                    if fx_restart < fx:
                        x, fx = x_restart, fx_restart
                    else:
                        # Try a point near the current best
                        x_restart = x + 0.5 * rng.standard_normal(dim)
                        x_restart = np.clip(x_restart, -5.0, 5.0)
                        fx_restart = safe_func(x_restart)
                        if fx_restart < fx:
                            x, fx = x_restart, fx_restart
                    
                    no_improve_count = 0
                    # Reset step sizes
                    sigma = np.ones(dim) * 1.0
            
            # Update global best
            if fx < best_f:
                best_f, best_x = fx, x.copy()
        
        # Final: try to polish with very small steps
        if call_count < budget and time.time() < deadline:
            sigma_small = np.ones(dim) * 0.1
            x_polish = best_x.copy()
            fx_polish = best_f
            for _ in range(min(20, budget - call_count)):
                if time.time() > deadline:
                    break
                y = x_polish + sigma_small * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = safe_func(y)
                if fy < fx_polish:
                    x_polish, fx_polish = y, fy
                    if fx_polish < best_f:
                        best_f, best_x = fx_polish, x_polish.copy()
                sigma_small *= 0.7
        
        return best_x, best_f
# EVOLVE-BLOCK-END
