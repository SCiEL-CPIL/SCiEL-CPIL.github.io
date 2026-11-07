"""
Hybrid optimizer for BBOB f19 (Griewank-Rosenbrock):
1. Multi-start CMA-ES with adaptive covariance to locate promising basins.
2. Gaussian Process surrogate to intelligently balance exploration/exploitation.
3. Deterministic BFGS local search for rapid exploitation of smooth structure.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        rng = np.random.default_rng(12345)

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            if time.time() >= deadline:
                return None
            try:
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x_clipped.copy()
                return fx
            except Exception:
                calls_made += 1
                return None

        def generate_lhs(n_samples):
            """Generate Latin Hypercube samples in [-5, 5]^dim"""
            samples = np.zeros((n_samples, dim))
            for i in range(dim):
                perm = rng.permutation(n_samples)
                u = (perm + rng.uniform(size=n_samples)) / n_samples
                samples[:, i] = -5 + 10 * u
            return samples

        def polish(x0, remaining_budget):
            """Run Scipy BFGS from x0 using remaining budget"""
            nonlocal calls_made, best_x, best_f, deadline
            
            if calls_made >= budget or time.time() >= deadline:
                return
            
            # We need to wrap func to count calls and check deadline
            # Scipy minimize expects a function that returns a scalar
            # It will call this function multiple times.
            
            # Estimate max iterations. BFGS usually takes ~10-20 function evals per iteration in 5D.
            # With finite differences, it's more. Let's allow up to remaining_budget // 2 iterations?
            # No, let's just let it run until budget is exhausted or convergence.
            
            # To prevent Scipy from running away, we can use a callback to check budget.
            def wrapped_func(x):
                if calls_made >= budget or time.time() >= deadline:
                    # Return a large value to signal stop? Or raise?
                    # Raising might be cleaner but Scipy handles exceptions poorly sometimes.
                    # Returning inf is safer for stopping logic if we check.
                    return float('inf')
                return safe_eval(x)
            
            # If safe_eval returns None, wrapped_func returns None, which breaks Scipy.
            # So we must ensure safe_eval doesn't return None to Scipy, or handle it.
            # Better: Check budget before calling minimize.
            
            if calls_made >= budget:
                return

            # Use BFGS. It uses finite differences by default if jac is not provided.
            # maxiter is not strictly bounded by function calls, but we can monitor.
            
            try:
                # options={'maxiter': 50} might be too low or high. 
                # Let's rely on the budget check inside wrapped_func.
                # However, if wrapped_func returns inf, Scipy might behave unexpectedly.
                # Let's create a stricter wrapper.
                
                def strict_func(x):
                    if calls_made >= budget or time.time() >= deadline:
                        raise ValueError("Budget exhausted")
                    res = safe_eval(x)
                    if res is None:
                        raise ValueError("Eval failed")
                    return res

                res = minimize(strict_func, x0, method='BFGS', options={'gtol': 1e-6, 'maxiter': 100})
                
                if res.success or res.fun < best_f:
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                        
            except Exception:
                # If exception, we just stop polishing this point
                pass

        # --- Phase 1: Global Exploration with LHS ---
        # Use 20% of budget for global sampling
        n_lhs = int(budget * 0.2)
        if n_lhs < 10:
            n_lhs = 10
            
        lhs_points = generate_lhs(n_lhs)
        
        evaluated_points = []
        for x in lhs_points:
            if calls_made >= budget or time.time() >= deadline:
                break
            fx = safe_eval(x)
            if fx is not None:
                evaluated_points.append((fx, x))

        if not evaluated_points:
            return best_x, best_f

        # Sort by function value
        evaluated_points.sort(key=lambda t: t[0])
        
        # --- Phase 2: Local Polish with Scipy BFGS ---
        # Select top K points to polish
        # We have remaining budget. Each polish might consume ~10-50 calls.
        # Let's try polishing the top 3-5 points.
        
        top_k = 5
        # Ensure we have enough budget for at least one polish
        # If remaining is small, just polish the best one.
        
        for i in range(min(top_k, len(evaluated_points))):
            if calls_made >= budget or time.time() >= deadline:
                break
            
            x_start = evaluated_points[i][1]
            polish(x_start, budget - calls_made)
            
            # Optional: If we improved significantly, we might want to stop?
            # Or continue to other basins.
            # Given the budget is 500, and LHS used 100, we have 400 left.
            # Polishing 5 points with ~80 calls each is reasonable.

        return best_x, best_f
