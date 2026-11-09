"""Rosenbrock Optimizer using Multi-Start L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """
    Optimizer for BBOB-style Rosenbrock (f8).
    Strategy: 
    1. Light global exploration to identify promising basins.
    2. Multi-start L-BFGS-B local optimization. L-BFGS-B is a quasi-Newton method 
       that approximates the Hessian, making it highly effective for the smooth, 
       curved valley of the Rosenbrock function. It converges much faster than 
       coordinate descent or Nelder-Mead for this specific problem structure.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        # State variables
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                raise StopIteration("Budget exhausted")
            if time.time() > deadline:
                raise StopIteration("Time limit exceeded")
            
            # Ensure x is a valid numpy array of correct shape and within bounds
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            calls_made += 1
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            
            return val

        # --- Phase 1: Define Starting Points ---
        # The minimizer is in [-4, 4]^5.
        # Standard Rosenbrock min is at (1, 1, ..., 1).
        # With shift s, min is at 1+s.
        # We use a mix of informed and random starts.
        
        rng = np.random.default_rng(42)
        
        starts = []
        # 1. Standard Rosenbrock minimum (highly likely if shift is small)
        starts.append(np.ones(dim))
        
        # 2. Origin
        starts.append(np.zeros(dim))
        
        # 3. Random points in the likely region [-3, 3]^5
        # This covers most of the shifted space efficiently.
        for _ in range(4):
            starts.append(rng.uniform(-3.0, 3.0, dim))
            
        # 4. A few points in the full box for robustness against large rotations/shifts
        for _ in range(2):
            starts.append(rng.uniform(-4.5, 4.5, dim))
            
        # Remove duplicates (approximate)
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5.0, 5.0)
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=1e-6):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)

        # --- Phase 2: Multi-Start L-BFGS-B Optimization ---
        # We iterate through starts, allocating remaining budget to each.
        # L-BFGS-B is very efficient, so we can afford multiple starts.
        
        for x0 in unique_starts:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            
            # If we have very little budget left, skip complex optimizers
            remaining_budget = budget - calls_made
            if remaining_budget < 10:
                continue

            try:
                # L-BFGS-B is suitable for bounded optimization
                # We set tight tolerances to exploit the budget for precision
                res = scipy.optimize.minimize(
                    safe_call, 
                    x0, 
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 200,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxfun': remaining_budget - 1 # Safety net, though safe_call handles it
                    }
                )
                
                # Update best if this run found something better
                # Note: safe_call already updates best_x/best_f internally
                # But we should ensure we are tracking the absolute best from scipy's final point
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
                    
            except StopIteration:
                # Budget or time exhausted during optimization
                break
            except Exception:
                # If an optimizer fails, continue to next start
                continue

        # Fallback if no valid calls were made or best_x is still None
        if best_x is None:
            x0 = np.ones(dim)
            try:
                best_f = safe_call(x0)
                best_x = x0.copy()
            except StopIteration:
                # Absolute last resort, though this shouldn't happen with 500 budget
                best_x = np.ones(dim)
                best_f = float('inf')

        return best_x, best_f

# EVOLVE-BLOCK-END
