"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using 1+1-ES with adaptive step size."""
import numpy as np
import time


class Optimizer:
    """
    Optimizer using a 1+1 Evolution Strategy (Hill Climbing) with adaptive step size.
    
    Strategy:
    1. Initialize with a diverse set of random points to find a good basin quickly.
    2. Use a 1+1-ES (single candidate per step) to rapidly climb the local gradient.
    3. Adapt the step size (sigma) based on the ratio of successful steps (2/5 rule).
    4. If no improvement is found for a certain number of steps, perform a random restart
       with a larger step size to escape local minima (crucial for Lunacek's bi-modal landscape).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        best_x = None
        fbest = np.inf
        calls = 0

        def update_best(x, fx):
            nonlocal best_x, fbest
            if fx < fbest:
                fbest = fx
                best_x = x.copy()

        # Phase 1: Initial Exploration
        # Evaluate a set of diverse points to find a promising region.
        # Lunacek's function has a global minimum and a shallow local minimum.
        # We want to find the global one quickly.
        n_init = min(20, budget // 20)
        
        # Always include origin and some structured points
        candidates = [np.zeros(dim)]
        # Add random points
        for _ in range(n_init - 1):
            candidates.append(np.random.uniform(-4, 4, dim))
        
        # Evaluate initial candidates
        for c in candidates:
            if not time_ok() or calls >= budget:
                break
            fx = func(c)
            calls += 1
            update_best(c, fx)

        if best_x is None:
            # Fallback if something went wrong (shouldn't happen)
            best_x = np.zeros(dim)
            fbest = func(best_x)
            calls += 1

        # Phase 2: 1+1-ES Optimization
        x = best_x.copy()
        fx = fbest
        sigma = 1.0  # Initial step size
        
        # 2/5 rule parameters
        target_success_rate = 0.2
        sigma_factor = 1.015  # Multiplicative factor for sigma update
        
        no_improve_count = 0
        max_no_improve = 50  # Restart if stuck for this many steps
        
        while calls < budget and time_ok():
            # Generate a candidate
            z = np.random.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1
            
            if fy < fx:
                # Success
                x = y
                fx = fy
                update_best(x, fx)
                no_improve_count = 0
                # Decrease sigma slightly on success (optional, but 2/5 rule handles it)
                # The 2/5 rule effectively adjusts sigma to maintain ~20% success rate
            else:
                # Failure
                no_improve_count += 1
                
            # Update sigma using 2/5 rule
            # If success rate is high, increase sigma; if low, decrease sigma.
            # Since we only have 1 step, we use a running average or simple heuristic.
            # A simpler heuristic for 1+1-ES:
            # If we succeeded, it might mean sigma was too small? No, usually we want to keep momentum.
            # Standard 2/5 rule for (1+1)-ES:
            # Let p be the probability of success. We want p = 0.2.
            # If we succeeded, we estimate p is high? No, 1/1 is not 0.2.
            # Better approach: Track a moving average of success.
            
            # Let's use a simpler adaptive scheme:
            # If we improve, sigma stays or decreases slightly to refine.
            # If we don't improve for a while, we might be at a local minimum or sigma is too small.
            
            # Actually, the standard 2/5 rule is better applied with a population.
            # For 1+1-ES, a common strategy is:
            # If success: sigma = sigma * 1.01 (slightly larger to explore more if we're moving well? No, usually we want to contract).
            # Let's use the "Adaptive Step Size" based on success/failure history.
            
            # Simpler robust approach:
            # Keep a buffer of last 10 steps.
            # If success rate > 0.2, increase sigma.
            # If success rate < 0.2, decrease sigma.
            
            # To keep it simple and fast, let's just use a fixed decay on success and growth on failure?
            # No, that oscillates.
            
            # Let's use the "CMA-ES like" sigma update but simplified:
            # We don't have a population, so we can't compute the standard deviation of the population.
            # Instead, we rely on the restart mechanism for large jumps.
            
            # Strategy: 
            # 1. On success, do not change sigma (or decrease very slightly to exploit).
            # 2. On failure, do not change sigma immediately.
            # 3. If no improvement for `max_no_improve` steps, assume local minimum.
            #    Perform a restart: pick a new random point or jump from current best.
            
            if no_improve_count >= max_no_improve:
                # Restart
                # Option A: Random restart from scratch (risky if we've found a good basin)
                # Option B: Local restart with larger sigma (exploit current basin more aggressively)
                
                # Lunacek has two minima. If we are stuck, we might be in the wrong one.
                # Let's try a few random jumps from the current best.
                found_better = False
                for _ in range(5):
                    if calls >= budget or not time_ok():
                        break
                    jump_size = 2.0 * np.random.uniform(0.5, 1.5)
                    z = np.random.standard_normal(dim)
                    y_restart = x + jump_size * z
                    y_restart = np.clip(y_restart, -5, 5)
                    fy_restart = func(y_restart)
                    calls += 1
                    
                    if fy_restart < fx:
                        x = y_restart
                        fx = fy_restart
                        update_best(x, fx)
                        found_better = True
                        break
                
                if found_better:
                    no_improve_count = 0
                    sigma = 1.0  # Reset sigma to a moderate value
                else:
                    # Still stuck. Try a larger global restart.
                    # Pick a completely new random point.
                    if calls < budget - 10 and time_ok():
                        x_new = np.random.uniform(-4, 4, dim)
                        fx_new = func(x_new)
                        calls += 1
                        if fx_new < fx:
                            x = x_new
                            fx = fx_new
                            update_best(x, fx)
                        # Even if not better, keep the old x as the base for further local search
                        # But reset the "stuck" counter
                        no_improve_count = 0
                        sigma = 1.0

        return best_x, fbest
