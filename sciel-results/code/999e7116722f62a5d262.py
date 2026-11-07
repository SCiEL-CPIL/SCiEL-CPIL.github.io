import numpy as np
import time

class Optimizer:
    """
    Hooke-Jeeves Pattern Search with adaptive restarts and time guards.
    Combines the structural coordinate-wise search of HJ with robust 
    step-size adaptation and stagnation handling.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, stop at 285s to be safe
        deadline = start_time + 285.0
        
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        eval_count = 0
        best_x = None
        best_f = float('inf')
        
        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initialization via random sampling
        # Try a few random points to get a good starting point
        # The Bent Cigar is separable, so random points are okay, 
        # but zero-centering is often good if the shift is small.
        # We will sample a mix of zero and random.
        
        candidates = []
        # 1. Zero
        candidates.append(np.zeros(dim))
        # 2. 10 random points
        for _ in range(10):
            if time.time() > deadline or eval_count >= budget:
                break
            candidates.append(rng.uniform(-5, 5, size=dim))
            
        # Evaluate candidates
        for x_c in candidates:
            if time.time() > deadline or eval_count >= budget:
                break
            f_c = eval_func(x_c)
        
        if best_x is None:
            # Fallback if no eval happened (shouldn't happen)
            best_x = np.zeros(dim)
            best_f = float('inf')

        # Phase 2: Hooke-Jeeves Optimization
        # Start from the best point found
        x = best_x.copy()
        fx = best_f
        
        # Initial step sizes: Moderate to allow exploration
        # Bent Cigar has a very narrow valley, so we need to be able to shrink fast
        # but also move fast along the wide axis.
        step_sizes = np.full(dim, 1.0)
        min_step = 1e-6
        max_step = 10.0
        
        # Stagnation handling
        stall_counter = 0
        stall_threshold = 50
        restart_count = 0
        max_restarts = 10
        
        # HJ Loop
        while eval_count < budget and time.time() < deadline:
            
            # If we have restarted too many times, break
            if restart_count > max_restarts:
                break
                
            moved = False
            # Iterate through dimensions
            for i in range(dim):
                if eval_count >= budget or time.time() > deadline:
                    break
                
                # Try positive step
                x_plus = x.copy()
                x_plus[i] += step_sizes[i]
                f_plus = eval_func(x_plus)
                
                if f_plus < fx:
                    x = x_plus
                    fx = f_plus
                    # Expand step size in this direction
                    step_sizes[i] = min(step_sizes[i] * 1.5, max_step)
                    moved = True
                    stall_counter = 0 # Reset stall on any move
                    # If we moved, we might want to check this dimension again
                    # but standard HJ moves to next. Let's stick to standard for simplicity
                    # and rely on the loop restart to re-check.
                    continue
                
                # Try negative step
                x_minus = x.copy()
                x_minus[i] -= step_sizes[i]
                f_minus = eval_func(x_minus)
                
                if f_minus < fx:
                    x = x_minus
                    fx = f_minus
                    # Expand step size in this direction
                    step_sizes[i] = min(step_sizes[i] * 1.5, max_step)
                    moved = True
                    stall_counter = 0
                    continue
                
                # No improvement in this dimension
                # Shrink step size
                step_sizes[i] = max(step_sizes[i] * 0.5, min_step)
            
            # End of dimension sweep
            if not moved:
                # No dimension could be improved
                # Check if steps are too small (converged)
                if np.all(step_sizes < 1e-4):
                    # Considered converged, try a restart with larger steps
                    # to escape potential local minima or scaling issues
                    pass 
                
                stall_counter += 1
                if stall_counter >= stall_threshold:
                    # Stagnation detected
                    restart_count += 1
                    # Reset step sizes to a larger value to allow exploration
                    # Keep x at the current best (which is global best so far)
                    step_sizes = np.full(dim, 2.0)
                    stall_counter = 0
                    # Optional: small random perturbation? 
                    # HJ is deterministic, so perturbation helps escape.
                    # But let's first try just resetting steps.
                    # If that fails, the next stall will trigger another restart.
                    
        return best_x, best_f
