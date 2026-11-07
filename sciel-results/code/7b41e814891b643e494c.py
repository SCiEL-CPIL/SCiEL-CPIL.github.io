"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid CMA-ES-like + Nelder-Mead with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s kill
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        
        # Phase 1: Random sampling to find a good starting point
        best_x = None
        best_f = np.inf
        n_calls = 0
        
        # Try a few random points and some structured ones
        candidates = []
        # Center
        candidates.append(np.zeros(dim))
        # Corners of [-4, 4]
        for sign in [-1, 1]:
            candidates.append(sign * 4 * np.ones(dim))
        # Random points
        for _ in range(20):
            candidates.append(rng.uniform(-4, 4, dim))
            
        for x in candidates:
            if not time_ok():
                break
            x = np.clip(x, -5, 5)
            fx = func(x)
            n_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # Phase 2: Nelder-Mead refinement from best point
        if time_ok() and best_x is not None:
            # Use scipy's minimize with Nelder-Mead
            # We need to wrap func to count calls and check time
            call_count = [0]
            
            def wrapped_func(x):
                call_count[0] += 1
                if call_count[0] > budget - n_calls:
                    return np.inf
                if not time_ok():
                    return np.inf
                return func(np.clip(x, -5, 5))
                
            try:
                res = minimize(wrapped_func, best_x, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: (1+1)-ES with adaptive sigma if time remains
        if time_ok() and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            max_es_calls = min(budget - n_calls - call_count[0] if 'call_count' in locals() else budget - n_calls, 200)
            
            for _ in range(max_es_calls):
                if not time_ok():
                    break
                if n_calls >= budget:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                n_calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    if sigma < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
