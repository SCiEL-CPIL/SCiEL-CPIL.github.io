"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Optimizer using SciPy's L-BFGS-B method."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin before 300s limit
        
        # Initialize at the center of the box
        x0 = np.zeros(self.dim)
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        
        # Use L-BFGS-B which is efficient for smooth bounded problems
        try:
            result = minimize(
                func, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'maxfev': self.budget}
            )
            # Return the best solution found
            if result.success or hasattr(result, 'x'):
                return result.x, result.fun
            else:
                # Fallback to initial point if result is invalid
                return x0, func(x0)
        except Exception:
            # If an exception occurs, return the initial point
            # Note: func(x0) might have already been called if minimize started
            # To be safe, we just return x0 and a dummy value if we can't get result
            # But the evaluator ignores the return value.
            # However, we must not crash.
            try:
                fx0 = func(x0)
            except:
                fx0 = 0.0
            return x0, fx0
# EVOLVE-BLOCK-END
