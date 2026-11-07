"""Black-box continuous optimizer for BBOB-style f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer using a hybrid strategy:
    1. Quick coarse sampling to find promising regions.
    2. Multi-start L-BFGS-B for rapid convergence.
    3. Iterative refinement with perturbations to escape local minima.
    
    The key insight for maximizing AOCC is that early progress is heavily weighted.
    L-BFGS-B is extremely efficient for smooth functions like BBOB f14, often
    converging in <20 function evaluations. By combining it with diverse starts
    and a tight budget management, we can achieve high precision quickly.
    
    Strategy:
    - Use a mix of Halton (low-discrepancy) and random points for starts.
    - Run L-BFGS-B from each start, stopping early if budget is tight.
    - If budget remains, perform local search around the best point with
      smaller perturbations to polish the solution.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin: 300s limit, 20s buffer
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

        def wrapped_func(x):
            nonlocal calls
            if calls >= budget:
                raise ValueError("Budget Exceeded")
            if time.time() > deadline:
                raise ValueError("Time Exceeded")
            calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)

        # Generate Halton sequence points for low-discrepancy coverage
        def halton_sequence(n, dims):
            points = []
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

        def map_to_box(points):
            return -5.0 + 10.0 * points

        rng = np.random.default_rng(12345)
        
        # Generate diverse starting points
        # 1. Halton sequence for uniform coverage
        n_halton = 15
        halton_points = halton_sequence(n_halton, dim)
        halton_mapped = map_to_box(halton_points)
        
        # 2. Random points for stochastic exploration
        n_random = 10
        random_points = rng.uniform(-5, 5, size=(n_random, dim))
        
        # 3. Origin and near-origin points (BBOB optima are often near center)
        origin = np.zeros(dim)
        near_origin = rng.uniform(-1, 1, size=(5, dim))
        
        # Combine all starts
        all_starts = np.vstack([halton_mapped, random_points, [origin], near_origin])
        
        # Shuffle to avoid bias in ordering
        indices = rng.permutation(len(all_starts))
        all_starts = all_starts[indices]
        
        bounds = [(-5.0, 5.0)] * dim
        
        # Phase 1: Multi-start L-BFGS-B
        # Each L-BFGS-B run typically uses 10-30 function evaluations.
        # With 500 budget, we can afford ~15-20 runs.
        for x0 in all_starts:
            if calls >= budget or not time_left():
                break
            
            try:
                # Use L-BFGS-B with tight tolerance for fast convergence
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 50, 'gtol': 1e-6}
                )
                
                if res.fun < best_f:
                    update_best(res.x, res.fun)
                    
            except (ValueError, Exception):
                # Budget or time exceeded
                if "Budget Exceeded" in str(e) or "Time Exceeded" in str(e):
                    break
                # Otherwise, continue to next start

        # Phase 2: Polish the best solution
        # If we have budget left, refine the best point found
        if calls < budget - 10 and time_left() and best_f < np.inf:
            # First, a tight polish
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except (ValueError, Exception):
                pass

        # Phase 3: Local search with perturbations to escape local minima
        if calls < budget - 50 and time_left() and best_f < np.inf:
            # Try several perturbations around the best point
            n_perturbations = 10
            for i in range(n_perturbations):
                if calls >= budget or not time_left():
                    break
                
                # Decreasing perturbation size
                scale = 2.0 / (i + 1)
                perturbation = rng.normal(0, scale, size=dim)
                x_perturbed = np.clip(best_x + perturbation, -5, 5)
                
                try:
                    res = minimize(
                        wrapped_func, 
                        x_perturbed, 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={'maxiter': 30, 'gtol': 1e-6}
                    )
                    if res.fun < best_f:
                        update_best(res.x, res.fun)
                except (ValueError, Exception):
                    break

        # Final polish if budget remains
        if calls < budget - 5 and time_left() and best_f < np.inf:
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 50, 'gtol': 1e-10}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except (ValueError, Exception):
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
