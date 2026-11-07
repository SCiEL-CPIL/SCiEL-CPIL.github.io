"""Rewritten optimizer for BBOB f17 (Schaffers F7) using Global-Local Switching strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global-Local Switching optimizer for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x_global = None
        best_f_global = np.inf
        total_calls = 0

        def evaluate(x):
            nonlocal total_calls, best_f_global, best_x_global
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            total_calls += 1
            if f < best_f_global:
                best_f_global = f
                best_x_global = x_clipped.copy()
            return f

        # Phase 1: Coarse grid search to identify best basin
        # Use a 3x3x3 grid in 5D (27 points) plus center and corners
        # Actually, for 5D, let's use a smarter approach:
        # Evaluate at origin, ±3 in each dimension, and some random points
        
        # First, evaluate at origin
        origin = np.zeros(dim)
        evaluate(origin)
        
        # Evaluate at ±2.5 in each coordinate direction (10 points)
        for d in range(dim):
            for sign in [-1, 1]:
                x = np.zeros(dim)
                x[d] = sign * 2.5
                evaluate(x)
        
        # Evaluate at ±4 in each coordinate direction (10 points)
        for d in range(dim):
            for sign in [-1, 1]:
                x = np.zeros(dim)
                x[d] = sign * 4.0
                evaluate(x)
        
        # Add some random points for diversity
        n_random = 5
        for _ in range(n_random):
            x = rng.uniform(-5, 5, dim)
            evaluate(x)
        
        # Now we have ~26 evaluations done. Find the best point.
        # Use this as the starting point for local optimization.
        
        if total_calls >= budget or time.time() > deadline - 10.0:
            return best_x_global, best_f_global
        
        # Phase 2: High-mutation-rate (1+1)-ES from the best point found
        x = best_x_global.copy()
        f = best_f_global
        
        # Initial step size: moderate to explore around the basin
        sigma = 1.0
        
        # (1+1)-ES with adaptive sigma
        # Use a simple adaptation rule: increase sigma on success, decrease on failure
        # With high mutation rate to escape local minima
        
        # Parameters for (1+1)-ES
        # Success probability for sigma adaptation
        # Standard: if improvement > threshold, increase sigma; else decrease
        
        # For Schaffers F7, we want aggressive exploration initially
        # Then gradually reduce as we converge
        
        # Track recent improvements to adapt sigma
        recent_improvements = []
        max_history = 20
        
        # Run (1+1)-ES until budget or time limit
        while total_calls < budget:
            if time.time() > deadline - 5.0:
                break
            
            # Generate candidate
            y = rng.standard_normal(dim)
            x_candidate = x + sigma * y
            
            # Evaluate
            f_candidate = evaluate(x_candidate)
            
            # Check for time/budget
            if total_calls >= budget or time.time() > deadline - 5.0:
                break
            
            # Accept if better
            if f_candidate < f:
                # Success
                x = x_candidate
                f = f_candidate
                recent_improvements.append(1)
                
                # Increase sigma on success (but cap it)
                sigma *= 1.05
                sigma = min(sigma, 5.0)
            else:
                # Failure
                recent_improvements.append(0)
                
                # Decrease sigma on failure
                sigma *= 0.95
                sigma = max(sigma, 1e-4)
            
            # Keep history bounded
            if len(recent_improvements) > max_history:
                recent_improvements.pop(0)
            
            # Check for stagnation: if no improvement in last max_history steps
            # and sigma is small, restart with larger sigma
            if len(recent_improvements) >= max_history:
                success_rate = sum(recent_improvements) / len(recent_improvements)
                if success_rate < 0.1 and sigma < 0.5:
                    # Stagnation detected: reset sigma and perturb position
                    sigma = 2.0
                    # Small random perturbation
                    x = x + 0.5 * rng.standard_normal(dim)
                    x = np.clip(x, -5, 5)
                    recent_improvements = []
        
        # Ensure at least one call was made
        if total_calls == 0:
            x = np.zeros(dim)
            fx = func(x)
            best_x_global = x
            best_f_global = fx

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
