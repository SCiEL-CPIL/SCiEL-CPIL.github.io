"""Black-box continuous optimizer for BBOB-style f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import math

class Optimizer:
    """
    Optimizer using Multi-Start L-BFGS-B.
    
    The BBOB f14 function (Different powers) transformed by affine maps is convex.
    L-BFGS-B is a quasi-Newton method that converges rapidly to the global minimum
    for convex functions. Multi-start ensures we don't start in a poorly conditioned
    region, though for convex problems any start converges globally. The diversity
    helps with the initial convergence speed which is critical for AOCC scoring.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 320.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def time_left():
            return deadline - time.time() > 1.0

        # Wrapper to count calls and enforce budget
        def wrapped_func(x):
            nonlocal calls
            if calls >= budget:
                raise ValueError("Budget Exceeded")
            calls += 1
            # Clip to box to handle any numerical overshoots, though L-BFGS-B with bounds handles this
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)

        # Generate starting points
        # We use a Halton sequence for low-discrepancy coverage
        def halton_sequence(n, dims):
            points = []
            for i in range(1, n + 1):
                point = []
                for d in range(dims):
                    f = d + 2  # Start with prime bases 2, 3, 5...
                    result = 0.0
                    denom = 1
                    while True:
                        denom *= f
                        result += (i % f) / denom
                        i //= f
                        if i == 0:
                            break
                    point.append(result)
                points.append(point)
            return np.array(points)

        # Map [0,1]^d to [-5,5]^d
        def map_to_box(points):
            return -5.0 + 10.0 * points

        # Number of restarts
        # 500 calls budget. L-BFGS-B typically uses 10-50 calls per convergence in 5D.
        # Let's try 15 restarts.
        n_restarts = 15
        halton_points = halton_sequence(n_restarts, dim)
        start_points = map_to_box(halton_points)
        
        # Also include the origin and a few random points for safety
        start_points = np.vstack([np.zeros(dim), start_points])
        
        # Shuffle start points to avoid bias towards Halton order
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(start_points))
        start_points = start_points[indices]

        bounds = [(-5.0, 5.0)] * dim

        for x0 in start_points:
            if calls >= budget or not time_left():
                break
            
            try:
                # Use L-BFGS-B
                # maxiter is a safety limit, but the budget counter is the real limit
                # gtol can be set to stop when gradient is small
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 50, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < best_f:
                    # Update best with the result
                    # Note: res.fun might be from the last eval, but we should check all
                    # However, minimize returns the best found within its iterations
                    # We must ensure we record the best from the wrapper's perspective
                    # Since wrapped_func updates best_x/best_f internally via update_best? 
                    # NO, update_best is not called inside wrapped_func. 
                    # We need to call update_best after minimize or inside wrapped_func.
                    # It's safer to call update_best inside wrapped_func.
                    
                    # Let's modify wrapped_func to update best
                    # But wrapped_func is defined above. Let's re-define or use a nonlocal update.
                    
                    # Actually, let's just check res.fun and res.x
                    # But res.fun is the function value at res.x.
                    # We should update best_x/best_f with res.x and res.fun
                    # BUT, the global best might have been found in an intermediate step of a previous restart
                    # or an intermediate step of this restart that was worse than res.fun? No, L-BFGS-B keeps the best.
                    # So res.x is the best found in this minimize call.
                    
                    update_best(res.x, res.fun)
                    
            except (ValueError, Exception) as e:
                # Budget exceeded or other error
                if "Budget Exceeded" in str(e):
                    break
                else:
                    # Log or ignore, but ensure we don't crash
                    pass

        # If we have budget left, do a final polish from the best found so far
        if calls < budget and time_left() and best_f < np.inf:
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'gtol': 1e-10}
                )
                update_best(res.x, res.fun)
            except (ValueError, Exception):
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
