"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """L-BFGS-B based optimizer with multi-restart strategy."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        best_x = None
        best_f = float('inf')
        calls_used = 0
        max_calls = budget

        def wrapped_func(x):
            nonlocal calls_used
            if calls_used >= max_calls:
                return best_f
            x_clipped = np.clip(x, -5.0, 5.0)
            val = float(func(x_clipped))
            calls_used += 1
            return val

        # Phase 1: Try several random restarts with L-BFGS-B
        # We'll use about 70% of budget for exploration/restarts
        # and 30% for final polishing if needed.
        
        # Strategy: 
        # - Use a fixed number of restarts (e.g., 5-8)
        # - Each restart uses a limited number of function evaluations
        # - Keep track of the best solution found
        
        # Let's allocate budget per restart
        # With 500 budget, let's do ~6 restarts with ~70 evals each,
        # leaving some buffer.
        
        num_restarts = 6
        budget_per_restart = (budget * 70) // 100  # 70% of budget for restarts
        budget_per_restart = max(budget_per_restart // num_restarts, 10)
        
        # But we need to be careful not to exceed total budget
        # Let's use a simpler approach: track calls globally
        
        for restart in range(num_restarts):
            if calls_used >= max_calls:
                break
            
            # Random starting point
            x0 = np.random.uniform(-5.0, 5.0, dim)
            f0 = wrapped_func(x0)
            calls_used += 0  # already counted in wrapped_func
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Run L-BFGS-B from this starting point
            # Limit the number of function evaluations for this run
            remaining = max_calls - calls_used
            if remaining <= 0:
                break
            
            # Use a reasonable maxfun for this sub-problem
            maxfun = min(remaining, budget_per_restart)
            
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxfun': maxfun, 'maxiter': maxfun, 'gtol': 1e-6}
                )
            except Exception:
                continue
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        
        # Phase 2: If we have budget left, do a final polish from best_x
        remaining = max_calls - calls_used
        if remaining > 5 and best_x is not None:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxfun': remaining, 'maxiter': remaining, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
