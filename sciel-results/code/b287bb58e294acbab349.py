"""Optimized program for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start (1+1)-ES with adaptive step size and stagnation-based restarts.
    Designed for BBOB f4 (Büche-Rastrigin) which is multimodal.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. We stop at 290s.
        deadline = start_time + 290.0 
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = None
        best_f = float('inf')
        evals_used = 0

        # Parameters for (1+1)-ES
        # Initial sigma should be large enough to cover the box [-5, 5]^d
        # Box width is 10. In d=5, diagonal is 10*sqrt(5) ~ 22.
        # Start sigma around 2.0-3.0 allows good exploration.
        initial_sigma = 2.5
        sigma_min = 1e-8
        sigma_max = 10.0
        
        # Stagnation parameters
        # If no improvement for this many evals, restart
        stagnation_threshold = 60 
        # How many restarts to allow? We have 500 evals.
        # Each restart costs 1 eval. The search phase costs up to stagnation_threshold.
        # So we can have roughly budget / (stagnation_threshold + 1) restarts.
        # 500 / 61 ~ 8 restarts.
        
        # We will loop until budget or time is up.
        # Each "phase" is a (1+1)-ES run starting from a new point.
        
        while evals_used < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Initialize a new trajectory
            # If we have a best_x, we might start near it with a large kick, 
            # or just random. Random is safer for global exploration in early phases.
            # However, starting near the best found so far with a large sigma is also good.
            # Let's use a mix: First few restarts random, later ones near best.
            
            if best_x is None:
                x = rng.uniform(-5, 5, dim)
            else:
                # 50% chance to random restart, 50% to perturb best
                if rng.random() < 0.5:
                    x = rng.uniform(-5, 5, dim)
                else:
                    # Large perturbation to escape local minimum
                    perturbation = 5.0 * rng.standard_normal(dim)
                    x = np.clip(best_x + perturbation, -5, 5)
            
            fx = func(x)
            evals_used += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = initial_sigma
            no_improve_count = 0
            phase_evals = 1
            
            # (1+1)-ES loop for this trajectory
            while evals_used < budget:
                if time.time() > deadline:
                    break
                
                # Generate offspring
                # Clip to box to ensure validity
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                evals_used += 1
                phase_evals += 1
                
                if fy < fx:
                    # Success
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    no_improve_count = 0
                    # Increase sigma
                    sigma *= 1.2
                else:
                    # Failure
                    no_improve_count += 1
                    # Decrease sigma
                    sigma *= 0.85
                
                # Clamp sigma
                sigma = max(sigma_min, min(sigma, sigma_max))
                
                # Stagnation check
                # If we haven't improved for a while, break out to restart
                if no_improve_count > stagnation_threshold:
                    break
                
                # If sigma becomes very small, we've likely converged to a local min
                # Break out to restart
                if sigma < 1e-5:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
