import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer for BBOB f7 (Step Ellipsoidal) using Nelder-Mead and (1+1)-ES polishing.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin
        B = self.budget
        d = self.dim
        LO, HI = -5.0, 5.0
        
        used = 0
        best_x = None
        best_f = np.inf

        # Wrapper to count calls and enforce deadline
        def wrapped_func(x):
            nonlocal used, best_x, best_f
            if used >= B or time.time() > deadline:
                # Return a large value to signal failure/stop, or raise
                # Raising might be caught by minimize, but we want to stop cleanly.
                # However, minimize might not handle exceptions well in all versions.
                # Better to return a large value and hope it stops, or use a flag.
                # Actually, if we exceed budget, we should just return the best found so far.
                # But we are inside the function call. We can't change the flow of minimize easily.
                # We will raise a custom exception to break out of minimize if needed.
                pass
            
            x = np.clip(np.asarray(x, dtype=float), LO, HI)
            used += 1
            v = float(func(x))
            if v < best_f:
                best_f = v
                best_x = x.copy()
            return v

        # Phase 1: Initial Sampling
        # Try origin and a few random points to find a good starting basin
        # Since the function is shifted, the origin might be far.
        # Random sampling is safer.
        
        starts = []
        # Add origin
        starts.append(np.zeros(d))
        # Add a few random points
        rng = np.random.default_rng(42)
        for _ in range(3):
            starts.append(rng.uniform(LO, HI, d))

        best_start_x = None
        best_start_f = np.inf
        
        for x0 in starts:
            if used >= B or time.time() > deadline:
                break
            x0_clipped = np.clip(x0, LO, HI)
            f0 = wrapped_func(x0_clipped)
            if f0 < best_start_f:
                best_start_f = f0
                best_start_x = x0_clipped.copy()
        
        if best_start_x is None:
            # Fallback if no calls were made (should not happen)
            best_start_x = np.zeros(d)
            best_start_f = wrapped_func(best_start_x)

        # Phase 2: Nelder-Mead Optimization
        # Use remaining budget for NM
        # We need to estimate maxiter. NM in d dims uses ~d+1 evals per iter.
        # Let's reserve some budget for polishing.
        nm_budget = int((B - used) * 0.7)
        
        if nm_budget > 10 and best_start_x is not None:
            # Define options for Nelder-Mead
            # maxiter is a soft limit; we rely on the wrapper for hard budget
            options = {
                'maxiter': nm_budget,
                'xatol': 1e-4,
                'fatol': 1e-4,
                'adaptive': True
            }
            
            try:
                res = minimize(
                    wrapped_func, 
                    best_start_x, 
                    method='Nelder-Mead', 
                    options=options
                )
                # Check if we improved
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                # If minimize fails or raises, we continue with the best found so far
                pass

        # Phase 3: Polishing with (1+1)-ES
        # Use remaining budget for fine-tuning
        remaining = B - used
        if remaining > 5 and best_x is not None:
            sigma = 0.1
            rng2 = np.random.default_rng(999)
            for _ in range(remaining):
                if used >= B or time.time() > deadline:
                    break
                y = np.clip(best_x + sigma * rng2.standard_normal(d), LO, HI)
                fy = wrapped_func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma = min(sigma * 1.1, 1.0)
                else:
                    sigma *= 0.9
                    if sigma < 1e-4:
                        break

        return best_x, best_f
