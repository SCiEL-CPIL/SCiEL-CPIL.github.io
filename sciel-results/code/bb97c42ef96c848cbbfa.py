"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid: CMA-ES style global search followed by L-BFGS-B local polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        
        # Wrap func to count calls and enforce budget
        call_count = 0
        max_calls = self.budget
        
        def counted_func(x):
            nonlocal call_count
            if call_count >= max_calls:
                raise RuntimeError("Budget exceeded")
            x = np.asarray(x, dtype=np.float64).ravel()
            x = np.clip(x, -5.0, 5.0)
            val = float(func(x))
            call_count += 1
            return val
        
        dim = self.dim
        rng = np.random.default_rng(42)
        
        # Phase 1: Global search using a population-based method
        # We'll use a simple (1+lambda)-ES with adaptive step size
        # to find a good starting point for local optimization.
        
        # Initial population
        pop_size = 10
        x_best = rng.uniform(-4, 4, dim)
        f_best = counted_func(x_best)
        sigma = 1.0
        
        # Global search phase: use ~60% of budget
        global_budget = int(0.6 * max_calls)
        
        for i in range(global_budget - 1):
            if time.time() > deadline:
                break
            
            # Generate offspring
            y = x_best + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = counted_func(y)
            
            if fy < f_best:
                x_best, f_best = y, fy
                sigma *= 1.1  # Increase step on success
            else:
                sigma *= 0.9  # Decrease step on failure
            
            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)
        
        # Phase 2: Local refinement with L-BFGS-B
        # Use remaining budget for local optimization
        remaining_calls = max_calls - call_count
        
        if remaining_calls > 10 and time.time() < deadline:
            try:
                # Use L-BFGS-B with finite differences
                # maxfun is the max number of function evaluations
                result = minimize(
                    counted_func,
                    x_best,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 200, 'maxfun': remaining_calls, 'maxls': 20}
                )
                if result.success or result.fun < f_best:
                    x_best = np.clip(result.x, -5.0, 5.0)
                    f_best = float(result.fun)
            except Exception:
                pass  # Keep the best from global search
        
        return x_best, f_best
# EVOLVE-BLOCK-END
