"""Hybrid Grid + L-BFGS-B optimizer for BBOB f20 (Schwefel)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid optimizer:
    1. Coarse random sampling to find a good initial point.
    2. Multiple L-BFGS-B local optimizations from diverse starting points.
    3. Final polish with small steps if budget remains.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls = 0

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        def safe_call(x):
            nonlocal calls
            if calls >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = func(x)
                calls += 1
                return fx
            except:
                return np.inf

        # Phase 1: Coarse Random Search (20% of budget)
        # This helps identify the general region of the global minimum
        # despite hidden shifts.
        n_phase1 = max(20, int(budget * 0.2))
        
        # We will store all evaluated points to pick the best few for local opt
        # But for efficiency, we just track the best and maybe a few sub-best
        # Actually, for L-BFGS-B, we need diverse starts.
        # Let's do 20 random points, then pick the best 5 for local opt.
        
        candidates = []
        for i in range(n_phase1):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.0, 4.0, dim)
            fx = safe_call(x)
            if np.isfinite(fx):
                candidates.append((fx, x))
                update_best(x, fx)

        if not candidates:
            # Fallback if something went wrong
            return best_x, best_f

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Local Optimization with L-BFGS-B
        # Pick top 5 diverse candidates. 
        # To ensure diversity, we might just take the top 1, and then
        # random points far from it? 
        # Simpler: Take top 5 from the coarse search.
        # If they are all in the same basin, we might miss others, but 
        # Schwefel is multimodal. 
        # Let's take top 1, and then 4 random points from the rest of the 
        # candidates that are somewhat far from the best.
        
        starts = []
        best_candidate_f, best_candidate_x = candidates[0]
        starts.append(best_candidate_x)
        
        # Add 4 more diverse starts
        # We can just take the next 4 best, or sample randomly from the remaining candidates
        # Let's take the next 4 best from the coarse search.
        for i in range(1, min(5, len(candidates))):
            starts.append(candidates[i][1])
        
        # If we have fewer than 5 candidates, pad with random points
        while len(starts) < 5 and calls < budget:
            x = rng.uniform(-4.0, 4.0, dim)
            # Evaluate to count call
            fx = safe_call(x)
            if np.isfinite(fx):
                update_best(x, fx)
            starts.append(x)

        # Run L-BFGS-B from each start
        # Budget remaining: budget - calls
        # L-BFGS-B uses function calls. We need to wrap func to count calls.
        # SciPy's minimize takes a function. We can't easily stop it mid-iteration
        # based on global call count without a custom callback or wrapper that raises.
        # Let's create a wrapper that counts calls and raises an exception if budget exceeded.
        # Then catch the exception.
        
        class BudgetExceeded(Exception):
            pass

        def wrapped_func(x):
            nonlocal calls
            if calls >= budget or time.time() > deadline:
                raise BudgetExceeded()
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = func(x)
                calls += 1
                return float(fx)
            except:
                return 1e10 # Return large value on error? Or raise?
                # Better to return a large value to let optimizer fail gracefully
                # but we need to count the call if it was made.
                # If func raises, it's an invalid result anyway? 
                # "an exception later keeps the calls made"
                # So if func raises, we should probably stop or handle it.
                # The task says "exception before the first call... invalid".
                # "exception later keeps the calls made".
                # So we should let the exception propagate? 
                # No, the Optimizer itself shouldn't crash if possible.
                # But if func crashes, we can't continue.
                # Let's assume func is well-behaved as per contract "returns a float".
                return 1e10

        # We need to be careful with the deadline.
        # L-BFGS-B might run for a long time.
        # We can pass a maxiter based on remaining budget.
        # Each iteration of L-BFGS-B uses ~dim function calls (finite diff) or 1 (analytic grad, which we don't have).
        # Since we don't have gradients, it uses finite differences.
        # So 1 iteration ~ dim calls.
        # Remaining budget R. Max iterations ~ R // dim.
        
        for x0 in starts:
            if calls >= budget or time.time() > deadline:
                break
            
            remaining = budget - calls
            if remaining < 10:
                break
                
            max_iter = max(1, remaining // (dim + 5)) # Approx
            
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': max_iter, 'maxfun': remaining, 'disp': False}
                )
                if res.success or res.status in [0, 1]: # 0: maxiter, 1: success
                    fx_res = res.fun
                    if np.isfinite(fx_res) and fx_res < best_f:
                        best_f = fx_res
                        best_x = res.x.copy()
            except BudgetExceeded:
                break
            except Exception:
                # If L-BFGS-B crashes for other reasons, continue to next start
                continue

        # Phase 3: Final Polish
        # If we have time left, do a few small random steps around best_x
        if best_x is not None and calls < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            # Use remaining budget for (1+1)-ES
            while calls < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = safe_call(y)
                if np.isfinite(fy):
                    if fy < fx:
                        x = y
                        fx = fy
                        update_best(x, fx)
                        sigma *= 1.05
                    else:
                        sigma *= 0.95
                    sigma = np.clip(sigma, 0.001, 1.0)
                else:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
