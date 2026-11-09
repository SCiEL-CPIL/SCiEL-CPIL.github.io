"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Coordinate Descent based optimizer for BBOB f24 (Lunacek bi-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        n = self.dim
        total_budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= total_budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            try:
                fx = float(func(x))
            except:
                fx = np.inf
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Initialize at origin (0,0,0,0,0) which is often close to the global minimum in Lunacek
        current_x = np.zeros(n)
        current_f = safe_func(current_x)
        
        if current_f == np.inf:
            return best_x, best_f

        # Initial step size: The domain is [-5, 5]. 
        # Lunacek's function has a specific structure. 
        # A step size of 1.0 is a reasonable start for coordinate descent on this scale.
        # We will decay it over time.
        step_size = 1.0
        decay_factor = 0.98 # Slow decay to allow exploration
        min_step = 1e-6
        
        # To handle the "bi-Rastrigin" part (two minima), we might need to jump.
        # But CD is local. 
        # Strategy: 
        # 1. Do CD for a while.
        # 2. If stuck (no improvement in last K steps), increase step size or jump.
        # Given the budget is 500, and CD is 1 call per axis step (if we do cyclic), 
        # one full cycle is 5 calls. 500/5 = 100 cycles.
        
        # We will implement a simple cyclic CD with adaptive step size.
        # If a step improves, keep direction and maybe increase step slightly (or keep).
        # If a step worsens, shrink step and try opposite direction? 
        # Or just shrink step and move to next axis.
        
        # Let's use a "hill climbing" CD:
        # For each axis i:
        #   Try x_i + step
        #   If better, accept.
        #   Else try x_i - step
        #   If better, accept.
        #   Else, shrink step for this axis (or globally) and move to next axis.
        
        # To be more efficient, we can use a fixed cyclic order but adapt the global step size.
        
        # Let's try a simpler approach: Randomized Coordinate Descent with Step Decay.
        # Pick a random axis (or cyclic), try +/- step.
        
        # Given the "axis-aligned" hint in the prompt (though Lunacek is rotated in BBOB, 
        # the prompt says "exploiting the axis-aligned nature..."), I will assume 
        # the prompt implies that CD works well here, possibly because the rotation 
        # is mild or the function structure allows CD to make progress.
        
        # Actually, BBOB f24 (Lunacek) is rotated. Pure CD might struggle.
        # However, the directive says: "Replace ... with a Coordinate Descent optimizer 
        # that performs cyclic, line-search-free steps ... exploiting the axis-aligned nature".
        # This suggests the specific instance might have properties where CD works, 
        # or the prompt is suggesting a heuristic that works surprisingly well.
        # I will follow the directive strictly.
        
        # Implementation details:
        # We have 500 calls.
        # We start with step_size = 2.0 (to cover the [-5,5] box quickly).
        # We perform cyclic updates.
        
        # To avoid getting stuck in the local minimum (which is at distance ~5.7 from global),
        # we might need a mechanism to escape. 
        # A simple way: if no improvement in last 20 calls, increase step size or jump.
        
        stagnant_count = 0
        last_best_f = current_f
        stagnation_threshold = 20
        
        # We'll use a simple loop
        # We track the step size per axis? Or global? 
        # Let's use global step size for simplicity, decaying it.
        
        # Actually, let's use a "Randomized Coordinate Descent" where we pick a random axis each time.
        # This is more robust to rotation than cyclic if the rotation is not aligned.
        # But the prompt says "cyclic". I will use cyclic.
        
        axis = 0
        
        while calls_made < total_budget and time.time() < deadline:
            # Try to move along axis `axis`
            # Step 1: Try positive step
            x_try = current_x.copy()
            x_try[axis] += step_size
            
            f_try = safe_func(x_try)
            
            if f_try == np.inf:
                break
                
            if f_try < current_f:
                # Accept
                current_x = x_try
                current_f = f_try
                stagnant_count = 0
                # Optionally increase step size slightly if we're making good progress
                # But keep it simple: just keep step size
            else:
                # Step 2: Try negative step
                x_try = current_x.copy()
                x_try[axis] -= step_size
                f_try = safe_func(x_try)
                
                if f_try == np.inf:
                    break
                    
                if f_try < current_f:
                    # Accept
                    current_x = x_try
                    current_f = f_try
                    stagnant_count = 0
                else:
                    # No improvement
                    stagnant_count += 1
                    # Shrink step size
                    step_size *= 0.9
                    if step_size < min_step:
                        step_size = min_step
                        
            # Move to next axis
            axis = (axis + 1) % n
            
            # Stagnation check
            if stagnant_count >= stagnation_threshold:
                # We are stuck. 
                # Option 1: Increase step size to jump over the barrier.
                # Option 2: Random jump.
                # Let's try increasing step size significantly to escape local minima.
                # Lunacek's local minimum is quite deep. 
                # If we are stuck, we are likely in a local minimum.
                # A random jump might be better.
                
                # Let's do a random jump within the box
                # But we need to be careful not to waste too many calls.
                # Let's just increase step size to 3.0 and reset stagnant count.
                # This allows larger moves.
                step_size = 3.0 
                stagnant_count = 0
                # Also, maybe shuffle the axis order? 
                # For simplicity, keep cyclic.
                
        # Final polish: 
        # If we have budget left, do a finer grid search or smaller steps around best_x
        # But CD with min_step 1e-6 should have converged if it found a minimum.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
