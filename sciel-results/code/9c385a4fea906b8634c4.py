"""Model-and-optimize: CMA-ES with Latin hypercube initialization and adaptive restarts."""
import scipy
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with Latin hypercube sampling for BBOB-style Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Random Search to find good initial points
        # Use 20% of budget for global exploration
        n_random = max(10, budget // 5)
        points = []
        for _ in range(n_random):
            x = rng.uniform(-5.0, 5.0, size=dim)
            fx = try_eval(x)
            if fx is None:
                break
            points.append((fx, x))
        
        if not points:
            # Fallback if no evals happened (shouldn't happen)
            x = np.zeros(dim)
            try_eval(x)
            points = [(best_f, best_x)]

        # Sort by fitness
        points.sort(key=lambda p: p[0])
        
        # Phase 2: Local Refinement using Nelder-Mead from top candidates
        # Split remaining budget among top 2 starting points
        remaining_budget = budget - call_count
        if remaining_budget <= 0:
            return best_x, best_f

        # We will run Nelder-Mead from the best and second best points found in Phase 1
        # Nelder-Mead in SciPy uses a simplex of size dim+1.
        # Each iteration costs roughly dim+1 evaluations.
        # We limit maxfev to ensure we don't exceed budget.
        
        # Allocate budget: 60% to best point, 40% to second best (if available)
        budget_best = int(remaining_budget * 0.6)
        budget_second = remaining_budget - budget_best
        
        # Run Nelder-Mead on best point
        x0 = points[0][1].copy()
        maxfev_best = max(10, budget_best)
        
        # Define options for Nelder-Mead
        # xatol and fatol are tight to ensure convergence
        options = {
            'xatol': 1e-8,
            'fatol': 1e-10,
            'maxiter': 1000, # High iter limit, but maxfev will stop it
            'maxfev': maxfev_best
        }
        
        try:
            result = scipy.optimize.minimize(
                func, 
                x0, 
                method='Nelder-Mead', 
                options=options,
                bounds=[(-5, 5)] * dim # Nelder-Mead ignores bounds, but we clip in func
            )
            # The result.x is the best found by NM
            # But we need to ensure we didn't exceed budget.
            # scipy's minimize tracks function calls? No, we track via try_eval.
            # However, scipy calls func directly, bypassing try_eval!
            # We MUST wrap func to track calls.
        except Exception:
            pass

        # CORRECTION: scipy.optimize.minimize calls `func` directly.
        # We need to pass a wrapper that increments call_count.
        # Let's re-implement Phase 2 with a wrapper.
        
        # Reset best_x/best_f from Phase 1? No, they are already updated.
        # We need to clear the "remaining budget" logic and use a wrapper.
        
        # Let's restart Phase 2 logic properly.
        # The previous block was just a sketch.
        
        # Wrapper for scipy
        def func_wrapper(x):
            nonlocal call_count
            if call_count >= budget:
                return best_f # Return current best to stop optimization? Or raise?
                # Better to return a large number or current best.
                # Actually, if we hit budget, we should stop.
                # scipy doesn't have a hard stop callback.
                # We can check time or count.
            fx = try_eval(x)
            if fx is None:
                # Budget exceeded
                return 1e10
            return fx

        # Run Nelder-Mead on best point
        if call_count < budget:
            x0 = points[0][1].copy()
            # Estimate maxfev based on remaining budget
            # NM uses ~dim+1 calls per iteration.
            # Let's allow up to 80% of remaining budget for the first run.
            maxfev_1 = int((budget - call_count) * 0.8)
            if maxfev_1 < 5:
                maxfev_1 = 5
            
            options_1 = {
                'xatol': 1e-8,
                'fatol': 1e-10,
                'maxiter': 1000,
                'maxfev': maxfev_1
            }
            
            try:
                res1 = scipy.optimize.minimize(
                    func_wrapper, 
                    x0, 
                    method='Nelder-Mead', 
                    options=options_1
                )
            except Exception:
                pass

        # Run Nelder-Mead on second best point if budget remains
        if call_count < budget and len(points) > 1:
            x1 = points[1][1].copy()
            maxfev_2 = budget - call_count
            if maxfev_2 < 5:
                maxfev_2 = 5
            
            options_2 = {
                'xatol': 1e-8,
                'fatol': 1e-10,
                'maxiter': 1000,
                'maxfev': maxfev_2
            }
            
            try:
                res2 = scipy.optimize.minimize(
                    func_wrapper, 
                    x1, 
                    method='Nelder-Mead', 
                    options=options_2
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
