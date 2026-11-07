"""Black-box continuous optimizer for BBOB-style f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer using Multi-Start L-BFGS-B with enhanced exploration.
    
    Strategy:
    1. Use a Halton sequence to generate diverse starting points.
    2. Apply L-BFGS-B (a gradient-based method) from each start point.
    3. L-BFGS-B converges rapidly for smooth functions like BBOB f14.
    4. This approach typically reaches high precision (low error) in fewer
       function evaluations than population-based methods like CMA-ES,
       which is critical for maximizing the AOCC score (early progress).
    5. Added random perturbations around the best point to escape local minima
       if L-BFGS-B gets stuck, and a final polish phase.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 300.0  # Safety margin before 330s kill
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
            return deadline - time.time() > 2.0

        # Wrapper to count calls and enforce budget
        def wrapped_func(x):
            nonlocal calls
            if calls >= budget:
                raise ValueError("Budget Exceeded")
            if time.time() > deadline:
                raise ValueError("Time Exceeded")
            calls += 1
            # Clip to box to handle any numerical overshoots
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)

        # Generate starting points using Halton sequence for low-discrepancy coverage
        def halton_sequence(n, dims):
            points = []
            # Prime bases for Halton sequence
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for i in range(1, n + 1):
                point = []
                for d in range(dims):
                    f = primes[d]
                    result = 0.0
                    denom = 1
                    ii = i
                    while True:
                        denom *= f
                        result += (ii % f) / denom
                        ii //= f
                        if ii == 0:
                            break
                    point.append(result)
                points.append(point)
            return np.array(points)

        # Map [0,1]^d to [-5,5]^d
        def map_to_box(points):
            return -5.0 + 10.0 * points

        # Number of restarts
        # L-BFGS-B is very efficient. 20 restarts * ~20-50 calls/restart = 400-1000 calls.
        # We have 500 calls. So ~10-20 restarts is appropriate.
        n_restarts = 20
        halton_points = halton_sequence(n_restarts, dim)
        start_points = map_to_box(halton_points)
        
        # Add origin and a few random points for robustness
        rng = np.random.default_rng(42)
        extra_points = np.vstack([np.zeros(dim), rng.uniform(-5, 5, size=(5, dim))])
        start_points = np.vstack([start_points, extra_points])
        
        # Shuffle start points to avoid bias
        indices = rng.permutation(len(start_points))
        start_points = start_points[indices]

        bounds = [(-5.0, 5.0)] * dim

        # Main optimization loop
        for x0 in start_points:
            if calls >= budget or not time_left():
                break
            
            try:
                # Use L-BFGS-B
                # maxiter is a safety limit, but the budget counter is the real limit
                # gtol 1e-8 is a good balance for BBOB precision
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'gtol': 1e-8}
                )
                
                # Update best with the result of this minimization
                # res.x is the best found in this specific run
                if res.fun < best_f:
                    update_best(res.x, res.fun)
                    
            except (ValueError, Exception) as e:
                # Budget exceeded or other error
                if "Budget Exceeded" in str(e) or "Time Exceeded" in str(e):
                    break
                else:
                    # Ignore other errors and continue to next start
                    pass

        # Final polish: If we have budget left, refine the best point found
        if calls < budget and time_left() and best_f < np.inf:
            try:
                # Try a few random perturbations around the best point to escape potential local minima
                # This is inspired by the hybrid approach, but we stick to L-BFGS-B for speed
                # We'll do a small local search with a few restarts around the best point
                
                # First, a standard polish
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 200, 'gtol': 1e-10}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
                
                # If we still have budget, try a few random perturbations
                if calls < budget - 50 and time_left():
                    for _ in range(5):
                        if calls >= budget or not time_left():
                            break
                        # Perturb the best point
                        perturbation = rng.normal(0, 0.5, size=dim)
                        x_perturbed = np.clip(best_x + perturbation, -5, 5)
                        
                        try:
                            res = minimize(
                                wrapped_func, 
                                x_perturbed, 
                                method='L-BFGS-B', 
                                bounds=bounds,
                                options={'maxiter': 50, 'gtol': 1e-8}
                            )
                            if res.fun < best_f:
                                update_best(res.x, res.fun)
                        except (ValueError, Exception):
                            break
                            
            except (ValueError, Exception):
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
