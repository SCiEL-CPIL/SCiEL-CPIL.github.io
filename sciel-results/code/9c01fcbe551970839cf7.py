"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Gradient-based optimizer for BBOB f5 (linear slope)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        def safe_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = np.clip(x, -5, 5)
            calls_used += 1
            try:
                f = float(func(x))
            except Exception:
                return np.inf
            if not np.isfinite(f):
                return np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # 1. Estimate the gradient using finite differences
        # Since f5 is linear, f(x) = g^T x + b.
        # We evaluate at 0 and at delta * e_i to find g_i.
        # Use a moderate delta to ensure we are in the linear regime (which is everywhere) 
        # and to avoid numerical cancellation if the function is very flat (it's not, slope is 2.5).
        delta = 1.0
        
        x0 = np.zeros(dim)
        f0 = safe_func(x0)
        best_f = f0
        best_x = x0.copy()

        if calls_used >= budget or time.time() > deadline:
            return best_x, best_f

        gradient = np.zeros(dim)
        for i in range(dim):
            if calls_used >= budget or time.time() > deadline:
                break
            xi = x0.copy()
            xi[i] += delta
            xi = np.clip(xi, -5, 5)
            fi = safe_func(xi)
            
            if np.isfinite(fi) and np.isfinite(f0):
                gradient[i] = (fi - f0) / delta
            else:
                # Fallback if evaluation failed
                gradient[i] = 0.0

        # 2. Determine the optimal corner
        # To minimize g^T x, we choose x_i = -5 if g_i > 0, and x_i = 5 if g_i < 0.
        # If g_i is 0, the direction doesn't matter, but we can pick 0 or keep current.
        # Since the minimum is on the boundary, picking the extreme is correct for linear functions.
        
        x_opt = np.zeros(dim)
        for i in range(dim):
            if gradient[i] > 0:
                x_opt[i] = -5.0
            elif gradient[i] < 0:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0
        
        x_opt = np.clip(x_opt, -5, 5)
        
        # 3. Evaluate the optimal corner
        f_opt = safe_func(x_opt)
        if f_opt < best_f:
            best_f = f_opt
            best_x = x_opt.copy()
            
        # 4. Refinement (Optional)
        # For a perfectly linear function, x_opt is the global minimum.
        # However, if there's any non-linearity or numerical noise, a few local steps might help.
        # But given the "Linear slope" description, it is likely strictly linear.
        # We can use remaining budget for a simple local search around x_opt if we want to be safe,
        # but it is likely unnecessary and might waste time if not careful.
        # Let's do a very quick Nelder-Mead style check if we have plenty of budget left,
        # but usually, the corner is the answer.
        
        # If we have significant budget left, let's just verify with a few random points near the corner
        # to ensure we didn't miss a non-linear feature (unlikely for f5).
        remaining_budget = budget - calls_used
        if remaining_budget > 10 and time.time() < deadline - 5:
             rng = np.random.default_rng(42)
             # Sample a few points in the neighborhood of the best solution
             for _ in range(min(10, remaining_budget)):
                 if time.time() > deadline:
                     break
                 # Small perturbation
                 x_check = best_x + rng.normal(0, 0.1, dim)
                 x_check = np.clip(x_check, -5, 5)
                 f_check = safe_func(x_check)
                 if f_check < best_f:
                     best_f = f_check
                     best_x = x_check.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
