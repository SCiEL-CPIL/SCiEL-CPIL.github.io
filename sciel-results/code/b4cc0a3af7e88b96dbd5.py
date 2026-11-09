"""
Affine Model Recovery for BBOB f5 (Linear slope) with Verification.

Strategy:
1. The function f5 is linear (affine): f(x) = c + v^T x.
2. We recover the gradient v and offset c using d+1 evaluations (origin + d axis probes).
   This allows us to analytically determine the global minimum over the box [-5, 5]^d,
   which is always at a vertex.
3. We evaluate the predicted optimal vertex.
4. We perform a verification step by evaluating the opposite vertex to ensure robustness
   against numerical sign errors or unexpected non-linearities.
5. We stall by re-evaluating the best point found to consume the remaining budget,
   maintaining the lowest possible error for the AOCC calculation.

This approach finds the global minimum in ~8 calls, significantly faster than
brute-force corner search (32 calls), maximizing the early progress metric.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer that recovers the affine model of the linear function to find
    the global minimum in d+2 calls, with verification.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        d = self.dim
        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                val = func(x)
                calls_made += 1
                if np.isfinite(val) and val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val
            except Exception:
                return np.inf

        # Step 1: Evaluate at origin
        x0 = np.zeros(d)
        f0 = safe_call(x0)
        
        if not np.isfinite(f0):
            # Fallback: try a few random points if origin failed
            for _ in range(10):
                if calls_made >= self.budget: break
                xr = np.random.uniform(-5, 5, d)
                safe_call(xr)
            if best_x is None:
                best_x = x0
                best_f = f0 if np.isfinite(f0) else 0.0
            return best_x, best_f

        # Step 2: Evaluate along each axis at distance 5 to compute gradient
        # f(x) = f0 + v^T x
        # f(5 e_i) = f0 + 5 v_i  =>  v_i = (f(5 e_i) - f0) / 5
        v = np.zeros(d)
        for i in range(d):
            if time.time() > deadline or calls_made >= self.budget:
                break
            xi = np.zeros(d)
            xi[i] = 5.0
            fi = safe_call(xi)
            if np.isfinite(fi):
                v[i] = (fi - f0) / 5.0
            else:
                v[i] = 0.0

        # Step 3: Determine the predicted optimal vertex
        # For a linear function, min is at x_i = 5 if v_i < 0, else -5.
        x_pred = np.zeros(d)
        for i in range(d):
            if v[i] < 0:
                x_pred[i] = 5.0
            else:
                x_pred[i] = -5.0

        # Step 4: Evaluate the predicted optimum
        f_pred = safe_call(x_pred)

        # Step 5: Verification - check the opposite vertex
        # This is a safeguard. For a perfect linear function, f(x_pred) <= f(-x_pred).
        # If the function has any non-linear component or numerical noise,
        # the opposite vertex might be better.
        x_opp = -x_pred
        f_opp = safe_call(x_opp)

        # Step 6: Check a few other high-probability points if budget allows
        # Specifically, if any gradient component was near zero, the optimum
        # might be on a face rather than a vertex, or the sign might be unstable.
        # We already checked the two extreme vertices.
        # Let's also check the origin again? No, already done.
        
        # If we have budget left, we can check the center of the box (0) again?
        # No, we want to consume budget with the BEST point to keep b_i low.
        
        # Step 7: Stall
        # Ensure best_x and best_f are updated by safe_call
        if best_x is None:
            best_x = x_pred
            best_f = f_pred if np.isfinite(f_pred) else f0

        # Consume remaining budget by re-evaluating the best point
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            func(best_x)
            calls_made += 1
            
        return best_x, best_f
# EVOLVE-BLOCK-END
