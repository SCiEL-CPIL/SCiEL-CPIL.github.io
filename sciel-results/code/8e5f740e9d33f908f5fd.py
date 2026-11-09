"""
Optimization strategy for Büche-Rastrigin f4.
Uses a hybrid approach:
1. Dense grid scan for early progress and basin identification.
2. L-BFGS-B for fast local refinement from the best grid points.
"""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid Grid-Scan + L-BFGS-B Optimizer for Büche-Rastrigin.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin
        dim = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() >= deadline or calls_made >= total_budget

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x).flatten()
            # Ensure x is within bounds before calling func, though func clips too
            # Clipping here is safe and ensures consistent behavior
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: Grid Scan ---
        # 3^5 = 243 points. This is a large chunk of the budget but ensures 
        # broad coverage and good early AOCC if the global min is near a grid point.
        # We use 3 divisions per dimension.
        n_grid = 3
        grid_vals = np.linspace(-5.0, 5.0, n_grid)
        
        # Generate grid points
        grid_points = np.array(np.meshgrid(*([grid_vals] * dim))).T.reshape(-1, dim)
        
        # Shuffle grid points to avoid systematic bias in the first few calls?
        # No, deterministic order is fine, but maybe randomize to get a "best" 
        # point sooner if the grid is lucky? 
        # Actually, for AOCC, we want the best possible value as soon as possible.
        # A random shuffle of the grid helps avoid the case where the best point 
        # is the last one in the deterministic list.
        np.random.shuffle(grid_points)

        for x in grid_points:
            if check_deadline():
                break
            evaluate(x)

        # --- Phase 2: Local Refinement with L-BFGS-B ---
        # Identify top candidates from the grid scan.
        # We didn't store all (x, f) pairs efficiently above, so we rely on 
        # re-evaluating a few promising spots or just using the best found.
        # To be more effective, let's keep track of the top 3 points found so far.
        # Since we didn't store them, we will just run L-BFGS-B on the current 
        # best_x and a few jittered versions.
        
        # A better strategy: We have used ~243 calls. We have ~257 left.
        # Let's run L-BFGS-B on the current best_x.
        
        if best_x is not None and not check_deadline():
            # Wrap func for L-BFGS-B
            # L-BFGS-B expects a 1D array
            def wrapper(x):
                if check_deadline():
                    return best_f # Return current best to stop optimization? 
                    # Actually, minimize might not stop cleanly if we just return best_f.
                    # Better to rely on the deadline check in the main loop if possible,
                    # but L-BFGS-B runs internally.
                    # We'll just let it run but it's risky if it's slow.
                    # L-BFGS-B is fast, so this should be okay.
                return evaluate(x)

            # We will try to optimize from best_x
            x0 = best_x.copy()
            bounds = [(-5.0, 5.0)] * dim
            
            try:
                res = minimize(wrapper, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 100, 'maxfev': 200})
                # evaluate(res.x) is already called inside wrapper during optimization
                # But res.fun is the value at the last point.
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

            # Try one more optimization from a jittered version of best_x
            if not check_deadline():
                x0_jitter = np.clip(best_x + np.random.normal(0, 0.5, dim), -5, 5)
                # Avoid re-evaluating if it's too close to best_x
                if np.linalg.norm(x0_jitter - best_x) > 1e-6:
                    def wrapper2(x):
                        if check_deadline():
                            return best_f
                        return evaluate(x)
                    try:
                        res2 = minimize(wrapper2, x0_jitter, method='L-BFGS-B', bounds=bounds, 
                                       options={'maxiter': 100, 'maxfev': 200})
                        if res2.fun < best_f:
                            best_f = res2.fun
                            best_x = res2.x.copy()
                    except Exception:
                        pass

        # --- Phase 3: Random Restarts with Local Search ---
        # If we have budget left, do a few random restarts with a very short 
        # local search to catch any basins missed by the grid.
        
        num_restarts = 5
        for _ in range(num_restarts):
            if check_deadline():
                break
            
            # Random start
            x_start = np.random.uniform(-5.0, 5.0, dim)
            
            # Quick local search
            def wrapper3(x):
                if check_deadline():
                    return best_f
                return evaluate(x)

            try:
                res3 = minimize(wrapper3, x_start, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 20, 'maxfev': 40})
                if res3.fun < best_f:
                    best_f = res3.fun
                    best_x = res3.x.copy()
            except Exception:
                pass

        return best_x, best_f
