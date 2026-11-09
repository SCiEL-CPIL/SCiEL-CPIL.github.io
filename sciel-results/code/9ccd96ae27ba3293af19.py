"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid Random Sampling + Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin well before 330s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Phase 1: Random Sampling to find a good starting point
        # We allocate 20% of the budget to initial exploration
        # Since Sphere is convex, any point in the basin leads to the same minimum,
        # but finding a point closer to the minimum helps Nelder-Mead converge faster
        n_sample = max(10, budget // 5)
        
        best_x = None
        best_f = np.inf
        
        # Include origin and some random points
        candidates = []
        candidates.append(np.zeros(dim))
        
        for _ in range(n_sample - 1):
            # Sample from a Gaussian centered at 0 with sigma=2, clipped to box
            # This biases towards the center where minima are often found
            x = rng.normal(0, 2, dim)
            x = np.clip(x, -5, 5)
            candidates.append(x)
        
        for x in candidates:
            if time.time() > deadline:
                break
            try:
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x
            except Exception:
                continue
        
        if best_x is None:
            # Fallback if all failed
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
            except Exception:
                return np.zeros(dim), 0.0

        # Phase 2: Nelder-Mead Local Optimization
        # Use the remaining budget for Nelder-Mead
        # Nelder-Mead is efficient for low-dimensional, smooth, non-linear functions
        # maxiter is set high, but we will monitor time and budget
        # Note: scipy's minimize counts function evaluations. 
        # We have used n_sample evaluations. We have budget - n_sample left.
        
        # Initial simplex size
        # A reasonable initial step size is 0.5 * range / dim or similar
        # Let's use 1.0 as a standard initial step
        initial_step = 1.0
        
        # We need to be careful with the number of evaluations Nelder-Mead uses.
        # By default, it might use more than our remaining budget if not constrained.
        # However, scipy's minimize doesn't have a direct "max_evals" stopper that is perfectly reliable 
        # without a callback. We will use a callback to stop if we exceed time or budget.
        
        evals_used = n_sample
        remaining_budget = budget - evals_used
        
        def callback(xk):
            nonlocal evals_used
            # scipy calls this after each iteration, but we want to count function evals.
            # Actually, scipy's minimize with method='Nelder-Mead' doesn't easily expose 
            # exact function eval count in the callback in a simple way without wrapping.
            # Let's wrap the function to count calls.
            pass

        # Better approach: Wrap the function to count calls and enforce limits
        func_calls = 0
        
        def wrapped_func(x):
            nonlocal func_calls
            if time.time() > deadline:
                raise TimeoutError()
            if func_calls >= remaining_budget:
                raise BudgetExhausted()
            func_calls += 1
            return func(x)

        class BudgetExhausted(Exception):
            pass

        # Run Nelder-Mead
        try:
            # initial_simplex: Nelder-Mead generates this from x0 and initial_step
            # We can provide a custom simplex for better control if needed, but default is okay
            res = minimize(
                wrapped_func, 
                best_x, 
                method='Nelder-Mead', 
                options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10, 'initial_simplex_size': 1.0}
            )
            if res.x is not None and res.fun < best_f:
                best_x = res.x
                best_f = res.fun
        except (TimeoutError, BudgetExhausted):
            pass
        except Exception:
            # If Nelder-Mead fails for any reason, keep the best from Phase 1
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
