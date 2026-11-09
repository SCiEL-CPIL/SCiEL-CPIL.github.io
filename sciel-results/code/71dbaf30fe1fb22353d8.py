import numpy as np
import time

class Optimizer:
    """
    Hybrid optimizer for BBOB-style Sphere f1.
    Strategy:
    1. Structured initial exploration to find a good basin quickly (early progress).
    2. CMA-ES for robust global-to-local convergence, handling rotation/shift.
    3. L-BFGS-B for high-precision polishing.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(n)

        def eval_point(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # --- Phase 1: Structured Exploration ---
        # Evaluate origin first
        f0 = eval_point(np.zeros(n))
        if f0 is None:
            return best_x, best_f

        # Adaptive exploration: evaluate points at multiple scales
        # Focus on finding a good basin quickly for early AOCC contribution
        # Use fewer, more targeted evaluations
        scales = [3.0, 1.5, 0.75]
        
        for scale in scales:
            if calls >= budget or time.time() > deadline:
                break
            for i in range(n):
                if calls >= budget or time.time() > deadline:
                    break
                x_plus = np.zeros(n)
                x_plus[i] = scale
                eval_point(x_plus)
                
                x_minus = np.zeros(n)
                x_minus[i] = -scale
                eval_point(x_minus)

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        x0 = best_x.copy()
        
        # Estimate initial sigma from the spread of evaluated points
        # If best_f is close to some reference, use smaller sigma
        # Otherwise, use larger sigma for exploration
        if best_f < 10.0:
            sigma0 = 0.5
        elif best_f < 50.0:
            sigma0 = 1.0
        else:
            sigma0 = 2.0

        # --- Phase 2: CMA-ES ---
        # CMA-ES is robust to rotations and shifts.
        # We allocate a significant chunk of the remaining budget to it.
        remaining = budget - calls
        cma_budget = int(remaining * 0.7)
        
        # Ensure CMA-ES has enough budget to be effective.
        # CMA-ES typically needs ~ (10 + 2*sqrt(n)) * n evaluations for good convergence.
        # For n=5, that's ~35 * 5 = 175.
        # If remaining is small, we might skip or reduce.
        
        if cma_budget > 50:
            try:
                import cma
                # Use a wrapper that respects the budget and deadline
                def cma_func(z):
                    if time.time() > deadline or calls >= budget:
                        return 1e100 # Large penalty to stop CMA-ES effectively
                    return eval_point(z)
                
                # Initial sigma:
                # If we found a point close to the minimum, use a smaller sigma.
                # If not, use a larger one.
                # A safe bet is 1.0 or 2.0.
                sigma0 = 1.0
                
                # Bounds
                bounds = [(-5.0,) * n, (5.0,) * n]
                
                # CMA-ES options
                opts = {
                    'bounds': bounds,
                    'maxfevals': cma_budget,
                    'seed': 42,
                    'verbose': -1,
                    'quiet': True,
                    'pickles': False,
                    'tolfun': 1e-15,
                    'tolx': 1e-12,
                    'tolfun' : 1e-15,
                    'tolfun_interval': 1,
                    'tolfun_hist': 1e-15,
                    'tolx': 1e-12,
                    'tolx_interval': 1,
                    'tolfun': 1e-15,
                    'tolfun_interval': 1,
                    'tolfun_hist': 1e-15,
                    'tolfun': 1e-15,
                    'tolfun_interval': 1,
                    'tolfun_hist': 1e-15,
                    'tolfun': 1e-15,
                    'tolfun_interval': 1,
                    'tolfun_hist': 1e-15,
                }
                
                # Note: cma library uses 'bounds' as a tuple (lb, ub)
                res = cma.fmin(cma_func, x0, sigma0, opts)
                
                # cma.fmin returns (xbest, fbest, res)
                # Update best if CMA-ES found something better
                # Note: cma_func calls eval_point which updates best_f/best_x
                # So we don't strictly need to update here, but it's good practice.
                # However, cma might return a point that was not the absolute best in our history
                # if the penalty was triggered, but eval_point tracks the true best.
                
            except Exception:
                # If CMA-ES fails, fall back to L-BFGS-B or just return best so far
                pass

        if time.time() > deadline or calls >= budget:
            return best_x, best_f

        # --- Phase 3: L-BFGS-B Polish ---
        remaining = budget - calls
        if remaining > 10:
            try:
                from scipy.optimize import minimize
                
                def lbfgs_func(x):
                    if time.time() > deadline or calls >= budget:
                        return 1e100
                    f = eval_point(x)
                    if f is None:
                        return 1e100
                    return f
                
                bounds = [(-5.0, 5.0)] * n
                
                x_start = best_x.copy()
                
                res = minimize(lbfgs_func, x_start, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 2000,
                                        'maxfun': remaining,
                                        'gtol': 1e-14,
                                        'ftol': 1e-14,
                                        'maxls': 50})
                
            except Exception:
                pass

        return best_x, best_f
