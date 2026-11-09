"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, deadline guard, and adaptive restart."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        d = self.dim
        N = self.budget
        rng = np.random.default_rng(0)
        
        # Global best tracking
        best_x = None
        best_f = np.inf
        
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        calls_made = 0
        
        # Parameters for (1+1)-ES
        sigma = 2.0
        sigma_min = 1e-6
        sigma_max = 10.0
        
        # Stagnation detection
        stagnation_counter = 0
        stagnation_threshold = 30  # Restart if no improvement for 30 steps
        
        # Initial point
        x = rng.uniform(-5, 5, d)
        fx = func(x)
        calls_made += 1
        best_x = x.copy()
        best_f = fx
        
        # Main loop
        while calls_made < N:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate trial point
            # Use a slightly larger step if stagnating to escape local minima
            # But keep it bounded
            effective_sigma = sigma
            if stagnation_counter > stagnation_threshold // 2:
                effective_sigma = sigma * 1.5
            
            y = x + effective_sigma * rng.standard_normal(d)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                # Improvement
                x = y
                fx = fy
                sigma *= 1.5
                sigma = min(sigma, sigma_max)
                stagnation_counter = 0
                
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # No improvement
                x = x  # Keep current
                sigma *= 0.87
                sigma = max(sigma, sigma_min)
                stagnation_counter += 1
                
                # Structural change: Adaptive Restart
                # If we are stuck, jump to a new random location, but keep the global best in mind.
                # However, pure random restart is bad for AOCC if we lose progress.
                # Instead, if stagnation is high, we can try a larger jump from the current best
                # or simply reset sigma and x to a new random point, but this risks losing the "early progress" benefit if we jump far.
                # A better strategy for AOCC: if stuck, do a local perturbation from the global best.
                # But since we are in (1+1), x is the current best.
                # Let's try: if stagnation_counter > threshold, reset x to a random point near the global best?
                # Or just increase sigma drastically?
                # The directive says: "Make a STRUCTURAL change... different optimization or search method".
                # The current method is (1+1)-ES. The inspiration was CMA-ES (scored poorly due to overhead/budget issues).
                # Let's stick to (1+1)-ES but improve the exploration/exploitation balance.
                # If stagnating, let's try a "hill climb" with larger steps or a random jump from the best.
                
                if stagnation_counter > stagnation_threshold:
                    # Restart mechanism:
                    # Option A: New random point. Risk: loss of AOCC progress if it's bad.
                    # Option B: Large jump from current best.
                    # Let's do a large jump from the current best (which is x).
                    # This is effectively what sigma increase does, but let's be more aggressive.
                    x = best_x + rng.standard_normal(d) * 2.0
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls_made += 1
                    if calls_made >= N:
                        break
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    else:
                        # If the jump was bad, revert to best_x and reduce sigma
                        x = best_x.copy()
                        fx = best_f
                    sigma = 1.0
                    stagnation_counter = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
