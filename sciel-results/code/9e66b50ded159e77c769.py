"""Combined (1+1)-ES with Nelder-Mead polish for BBOB-style Bent Cigar (f12)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with adaptive sigma followed by Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        # Reserve budget for Nelder-Mead polish
        # Nelder-Mead in dim=5 uses roughly dim+1 = 6 points, 
        # each iteration might call func ~2-4 times on average.
        # Let's reserve 100 calls for polish to allow ~20-30 iterations.
        polish_budget = min(100, budget // 5)
        es_budget = budget - polish_budget

        # Initial point: Random uniform in [-5, 5]
        # The minimizer is shifted, so random start is safer than origin
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_made = 1

        # (1+1)-ES Parameters
        sigma = 3.0  # Initial step size
        success_rate = 1/5  # 1/5 rule parameter, but we use simple multiplicative updates
        
        # Phase 1: (1+1)-ES
        while calls_made < es_budget and time.time() < deadline:
            # Generate candidate
            z = rng.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Success
                x, fx = y, fy
                sigma *= 1.2
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                # Failure
                sigma *= 0.9

            # Clamp sigma
            if sigma > 15.0:
                sigma = 15.0
            elif sigma < 1e-12:
                sigma = 1e-12

        # Phase 2: Nelder-Mead Polish
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and time.time() < deadline:
            nm_calls = [0]
            
            def nm_func(z):
                if time.time() >= deadline:
                    raise Exception("timeout")
                if nm_calls[0] >= remaining_budget:
                    raise Exception("budget exceeded")
                nm_calls[0] += 1
                # Clip to box to ensure validity
                z_clipped = np.clip(z, -5, 5)
                return func(z_clipped)
            
            try:
                result = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining_budget, 
                        'xatol': 1e-8, 
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if result.fun < best_fx:
                    best_x = result.x
                    best_fx = result.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
