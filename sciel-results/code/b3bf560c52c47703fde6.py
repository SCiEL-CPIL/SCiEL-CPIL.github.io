"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

class Optimizer:
    """Multi-start Powell optimizer with strict budget control."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin
        
        # We will perform multiple restarts.
        # Dim is 5. Powell is efficient.
        # Let's use 10 restarts, 50 evaluations each.
        n_restarts = 10
        budget_per_restart = self.budget // n_restarts
        
        best_x = None
        best_f = np.inf
        
        rng = np.random.default_rng(42) # Fixed seed for reproducibility of restarts
        
        # Generate restart points
        # Use a mix of random points and some structured points (e.g., corners, center)
        starts = []
        # Center
        starts.append(np.zeros(self.dim))
        # Random points
        for _ in range(n_restarts - 1):
            starts.append(rng.uniform(-5, 5, self.dim))
            
        # Shuffle starts to ensure diversity in early progress if some fail
        rng.shuffle(starts)
        
        total_calls = 0
        
        for x0 in starts:
            if total_calls >= self.budget:
                break
            if time.time() > deadline:
                break
                
            # Define a wrapper to count function calls
            call_count = 0
            def wrapped_func(x):
                nonlocal call_count
                call_count += 1
                total_calls += 1
                # Clip to box as per task description (func does this, but good to be safe)
                x_clipped = np.clip(x, -5, 5)
                return func(x_clipped)
                
            # Run Powell optimization
            # Powell uses line searches, which fits the directive
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Powell',
                    options={'maxiter': 50, 'maxfev': budget_per_restart, 'xtol': 1e-8, 'ftol': 1e-8}
                )
            except Exception:
                # If an exception occurs, we keep the calls made so far
                pass

            if res is not None and res.fun < best_f:
                best_f = res.fun
                best_x = res.x

        # If no optimization ran or failed, return the best initial point found?
        # The task says "return value of __call__ is ignored" but we should return something valid.
        # Actually, the evaluator ignores the return value, but we must not crash.
        if best_x is None:
             # Fallback: return center
             best_x = np.zeros(self.dim)
             best_f = func(best_x)
             
        return best_x, best_f
# EVOLVE-BLOCK-END
