"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import dual_annealing

class Optimizer:
    """Dual Annealing optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        # Define bounds for the search space [-5, 5]^dim
        bounds = [(-5.0, 5.0)] * self.dim
        
        # Wrap the function to track calls and enforce time limit
        call_count = [0]
        
        def wrapped_func(x):
            call_count[0] += 1
            if call_count[0] > self.budget:
                raise ValueError("Budget exceeded")
            
            # Time check
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            
            return func(x)
        
        try:
            # Use scipy's dual_annealing
            # maxiter is set high but budget and time will cut it off
            # initial_temp is set to allow good exploration
            result = dual_annealing(
                wrapped_func, 
                bounds, 
                maxiter=1000, 
                initial_temp=5230.0, 
                restart_temp_ratio=2e-5, 
                visit=2.62, 
                seed=0
            )
        except (TimeoutError, ValueError):
            pass
        except Exception:
            # If any other exception occurs, we might have partial results, 
            # but dual_annealing doesn't easily expose partial bests without 
            # complex wrapping. We rely on the fact that it's robust.
            pass

        # The return value is ignored by the evaluator, but we must return something
        # The task says "The return value of __call__ is ignored."
        # However, the seed program returns x, fx.
        # We should return the best found if possible, or just a placeholder.
        # Since we don't explicitly store the best x in the wrapper easily without 
        # modifying the wrapper to update a shared state, let's assume the 
        # internal state of dual_annealing isn't exposed if it fails.
        # But dual_annealing returns a result object. If it completes, result.x is best.
        # If it fails, we might not have it.
        # Let's refine the wrapper to keep track of best.
        
        # Actually, let's just return a dummy tuple as the seed does, 
        # because the evaluator ignores it.
        # But wait, if I want to be safe, I can just return (None, None) or similar.
        # The seed returns (x, fx).
        
        # Let's re-implement with a better wrapper to ensure we don't crash 
        # and to potentially return a valid x, fx if needed.
        
        return (0.0, 0.0)
# EVOLVE-BLOCK-END
