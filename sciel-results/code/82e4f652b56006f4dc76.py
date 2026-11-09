"""Portfolio-based search for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start (1+1)-ES with plateau-aware adaptation for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit. 280s is a safe margin to ensure we return before kill.
        deadline = start_time + 280.0
        
        d = self.dim
        budget = self.budget
        
        # Global best tracking
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # Random number generator
        rng = np.random.default_rng(42)
        
        # (1+1)-ES parameters tuned for d=5 and f7
        # Lambda is effectively 1 (one offspring)
        # Initial sigma
        sigma_init = 2.0
        
        # Restart logic
        # We split the budget into a few phases or just run until budget/time is up.
        # For f7, restarting is crucial to escape plateaus.
        # Strategy: 
        # 1. Start with a random point.
        # 2. Run (1+1)-ES.
        # 3. If no improvement for N generations, restart from a new random point 
        #    but keep the global best.
        # 4. To maximize AOCC (area under curve), we want to make progress early.
        #    So we should do a quick global search first, then refine.
        
        # Phase 1: Global search to find a good basin
        # Use a small number of random restarts to find the best starting point.
        # Since f7 has steps, a random start might be in a high plateau.
        # We want to find the low plateau (the minimum region) quickly.
        
        # Let's reserve 20% of the budget for initial global exploration.
        n_init = max(5, budget // 10)
        
        for i in range(n_init):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break
                
            x = rng.uniform(-4.5, 4.5, d)
            fx = func(x)
            calls_used += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            # Fallback if no calls were made (should not happen)
            x = np.zeros(d)
            best_f = func(x)
            calls_used += 1
            best_x = x.copy()
            
        # Phase 2: Local refinement with restarts
        # We will run (1+1)-ES starting from the current best.
        # If it stagnates, we restart from a new random point, 
        # but we will bias the new random point to be near the current best 
        # if the current best is already quite good (close to 0 error), 
        # or purely random if it's still far.
        
        # Actually, for f7, the "step" is the main issue. 
        # Once you cross the step, the function is ellipsoidal.
        # The minimum is at the center of the ellipsoid.
        
        # Parameters for (1+1)-ES
        c_sigma = 1.0 / (d + 3)  # Step size adaptation rate
        c_c = 4 / (d + 4)        # Not used in standard (1+1) but good for reference
        # For (1+1)-ES, we adapt sigma based on success rate.
        # Standard (1+1)-ES for CMA-ES like behavior:
        # sigma *= exp(c_sigma * (success - 1/5)) where success is 1 if improved, 0 otherwise?
        # Actually, a simpler plateau-aware strategy:
        # If improvement: sigma *= 1.1 (or similar)
        # If no improvement: sigma *= 0.9
        # If no improvement for K steps: Restart
        
        # Let's use a more robust sigma adaptation:
        # sigma_{t+1} = sigma_t * exp(c_sigma * (s_t - 1/5))
        # where s_t is 1 if f(x_{t+1}) < f(x_t) else 0.
        # This is the 1/5th rule.
        
        # However, for plateaus, f(x_{t+1}) == f(x_t) is possible.
        # In f7, plateaus are flat. So f(x_{t+1}) == f(x_t) is common.
        # We should treat "no improvement" as failure, but maybe less severe?
        # Or, detect plateau and increase sigma drastically to jump out.
        
        # Let's implement a hybrid:
        # 1. Track stagnation counter.
        # 2. If stagnation > threshold, increase sigma significantly (jump).
        # 3. If still stuck after several jumps, restart from random.
        
        stagnation_threshold = 10
        jump_factor = 2.0
        restart_threshold = 50  # If we've been stuck for 50 steps, restart completely
        
        current_x = best_x.copy()
        current_f = best_f
        sigma = sigma_init
        stagnation = 0
        
        while calls_used < budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            z = rng.standard_normal(d)
            y = current_x + sigma * z
            y = np.clip(y, -5.0, 5.0)
            
            fy = func(y)
            calls_used += 1
            
            # Update global best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                
            # Adapt sigma
            if fy < current_f:
                # Improvement
                current_x = y.copy()
                current_f = fy
                stagnation = 0
                # Increase sigma slightly to keep moving
                sigma *= 1.1
            else:
                # No improvement (includes plateau)
                stagnation += 1
                # Decrease sigma to explore locally? 
                # Or if on plateau, we might want to increase to jump out.
                # For f7, if we are on a plateau, decreasing sigma keeps us on the plateau.
                # Increasing sigma helps jump to a lower plateau.
                # So, on plateau (fy == current_f), we should increase sigma.
                # On local minimum (fy > current_f), we should decrease sigma.
                
                if fy == current_f:
                    # Plateau: increase sigma to jump
                    sigma *= 1.2
                else:
                    # Local hill: decrease sigma
                    sigma *= 0.9
                    
            # Clamp sigma
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Restart logic
            if stagnation > restart_threshold:
                # Restart from a new random point
                # But keep the global best.
                # New start point: random
                new_x = rng.uniform(-4.5, 4.5, d)
                new_f = func(new_x)
                calls_used += 1
                
                if new_f < best_f:
                    best_f = new_f
                    best_x = new_x.copy()
                
                # If the new point is better than the current stuck point, move there
                # Otherwise, we might stay stuck? 
                # Actually, if we restart, we should reset the local state.
                # Let's always restart to a random point if stuck.
                current_x = new_x.copy()
                current_f = new_f
                sigma = sigma_init
                stagnation = 0
                
            # If we are very close to the global best and stagnating, 
            # maybe we should try a random restart biased towards the best?
            # No, pure random restarts are safer for f7's multiple basins.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
