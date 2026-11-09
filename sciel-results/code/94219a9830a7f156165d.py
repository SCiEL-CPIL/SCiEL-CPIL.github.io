"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid optimizer: Adaptive 1+1 Evolution Strategy followed by Nelder-Mead polish.
    Designed for the Bent Cigar function (BBOB f12).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        dim = self.dim
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_fx = float('inf')
        calls_used = 0

        def check_time():
            return time.time() < deadline

        def make_call(x):
            nonlocal calls_used
            if not check_time():
                return None
            fx = func(x)
            calls_used += 1
            return fx

        # Phase 1: Coarse Adaptive Random Walk (1+1 ES)
        # Start with a reasonable sigma
        x = rng.uniform(-5, 5, dim)
        fx = make_call(x)
        if fx is None:
            return best_x, best_fx
        best_x = x.copy()
        best_fx = fx
        calls_used = 1

        sigma = 2.0
        successes = 0
        total = 0
        success_threshold = 0.2
        sigma_min = 1e-6
        sigma_max = 10.0

        # Run coarse search for ~60% of the budget or until converged
        coarse_budget = int(self.budget * 0.6)
        
        while calls_used < coarse_budget and check_time():
            direction = rng.standard_normal(dim)
            y = np.clip(x + sigma * direction, -5, 5)
            fy = make_call(y)
            if fy is None:
                break
            
            if fy < fx:
                x = y
                fx = fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                successes += 1
            total += 1

            if total >= 5:
                rate = successes / total
                if rate > success_threshold:
                    sigma *= 1.3
                elif rate < success_threshold:
                    sigma *= 0.7
                successes = 0
                total = 0

            sigma = np.clip(sigma, sigma_min, sigma_max)

            # If sigma is very small, we might be converged or stuck
            if sigma < 0.05:
                break

        # Phase 2: Nelder-Mead Polish
        # Use the best point found so far as the starting point for NM
        if check_time() and calls_used < self.budget:
            # Nelder-Mead can be sensitive to initial simplex size
            # Bent Cigar is ill-conditioned, so we use a moderate initial size
            # The function clips to [-5, 5], so we ensure initial points are within bounds
            x0 = best_x.copy()
            
            # We need to wrap the func to count calls and handle bounds
            # scipy minimize does not handle box constraints natively in Nelder-Mead
            # We rely on the internal clipping of the provided func(x)
            
            def wrapper(x):
                if not check_time():
                    return 1e10 # Return large value to stop
                if calls_used >= self.budget:
                    return 1e10
                val = func(x)
                calls_used += 1
                return val

            # Initial simplex: x0 and x0 + delta * e_i
            # Delta should be small enough to stay in basin, large enough to see gradient
            delta = 0.5 
            # If best_x is near boundary, adjust delta
            # For simplicity, use a fixed delta and let clipping handle it
            # However, NM might wander out. The wrapper's clipping in func(x) (task says func clips)
            # ensures we don't go out of bounds, but NM might get stuck on the boundary.
            
            try:
                res = minimize(
                    wrapper, 
                    x0, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 100, # Limit iterations to save budget
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if res.success and res.fun < best_fx:
                    best_x = res.x
                    best_fx = res.fun
            except Exception:
                pass

        # Phase 3: Final Fine Tuning (if budget remains)
        if check_time() and calls_used < self.budget:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.05
            successes = 0
            total = 0
            
            while calls_used < self.budget and check_time():
                direction = rng.standard_normal(dim)
                y = np.clip(x + sigma * direction, -5, 5)
                fy = make_call(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    successes += 1
                total += 1

                if total >= 5:
                    rate = successes / total
                    if rate > success_threshold:
                        sigma *= 1.2
                    elif rate < success_threshold:
                        sigma *= 0.7
                    successes = 0
                    total = 0

                sigma = np.clip(sigma, 1e-8, 1.0)
                if sigma < 1e-4:
                    break

        return best_x, best_fx

# EVOLVE-BLOCK-END
