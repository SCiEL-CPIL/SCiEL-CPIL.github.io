"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start Randomized Coordinate Descent for BBOB-style Sphere f1."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard: Evaluator kills at 330s. We aim for 300s hard limit.
        # Leave 10s buffer for safety.
        t0 = time.time()
        deadline = t0 + 300.0

        rng = np.random.default_rng(12345)
        
        # Global best tracking
        global_best_x = None
        global_best_f = float('inf')
        
        calls_made = 0
        
        # Strategy: Multiple restarts.
        # Each restart uses a portion of the budget.
        # We will run restarts until budget is exhausted or time is up.
        
        # Number of restarts? 
        # For Sphere, RCD converges fast. 
        # Let's use a dynamic approach: 
        # Start with a few random points, then refine.
        # Actually, a simple multi-start with equal budget allocation is robust.
        # Let's allocate budget per restart.
        # If we have 500 calls and 5 dims.
        # RCD typically needs ~ 10-50 * dim calls to converge well.
        # 50 * 5 = 250 calls per restart.
        # So we can afford 2 restarts with 250 calls each.
        # Or 5 restarts with 100 calls each.
        # More restarts = better chance of finding the global minimum early.
        
        # Let's try 5 restarts, 100 calls each.
        num_restarts = 5
        calls_per_restart = budget // num_restarts
        
        # If budget is small, do fewer restarts
        if calls_per_restart < 20:
            num_restarts = 1
            calls_per_restart = budget
            
        for r in range(num_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            
            # Initialize start point for this restart
            # Use a random point in the box
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()
            
            # Randomized Coordinate Descent
            # We maintain a history of directions to avoid repeating axes if possible,
            # but pure random axis selection is standard RCD.
            
            # To improve AOCC, we want to make quick initial progress.
            # A simple local search with adaptive step size helps.
            
            # Initial step size
            step = 1.0 
            
            # Local budget for this restart
            local_calls = 0
            max_local_calls = calls_per_restart - 1 # 1 call used for init
            
            while local_calls < max_local_calls:
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Choose a random coordinate
                i = rng.integers(0, dim)
                
                # Try to move in positive direction
                x_trial = x.copy()
                x_trial[i] += step
                # Clip to box
                if x_trial[i] > 5: x_trial[i] = 5
                if x_trial[i] < -5: x_trial[i] = -5
                
                f_trial = func(x_trial)
                calls_made += 1
                local_calls += 1
                
                if f_trial < fx:
                    x = x_trial
                    fx = f_trial
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x.copy()
                    # Increase step slightly if we made good progress? 
                    # Or keep it constant? 
                    # For Sphere, optimal step depends on curvature.
                    # Let's try a simple adaptation: if we improved, keep step.
                    # If we didn't, reduce step.
                else:
                    # Try negative direction
                    x_trial = x.copy()
                    x_trial[i] -= step
                    if x_trial[i] > 5: x_trial[i] = 5
                    if x_trial[i] < -5: x_trial[i] = -5
                    
                    f_trial = func(x_trial)
                    calls_made += 1
                    local_calls += 1
                    
                    if f_trial < fx:
                        x = x_trial
                        fx = f_trial
                        if fx < global_best_f:
                            global_best_f = fx
                            global_best_x = x.copy()
                    else:
                        # No improvement in either direction. Reduce step size.
                        step *= 0.5
                        # If step becomes too small, we are likely at a local min (or global for sphere)
                        if step < 1e-4:
                            break
                        
            # End of restart loop
            
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
