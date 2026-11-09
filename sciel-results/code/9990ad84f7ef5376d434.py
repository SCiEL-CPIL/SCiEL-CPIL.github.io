# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Grid-scan followed by deterministic local refinement for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls
            if calls >= self.budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            return fx

        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse Grid Scan
        # Use 3 points per dimension: -4, 0, 4
        # Total points: 3^5 = 243
        grid_points = [-4.0, 0.0, 4.0]
        
        # Generate grid
        grid = np.array(np.meshgrid(*([grid_points] * dim))).T.reshape(-1, dim)
        
        # Shuffle grid to ensure early progress score (AOCC) benefits from random-ish order
        perm = rng.permutation(len(grid))
        grid = grid[perm]

        for point in grid:
            if calls >= budget or time.time() > deadline:
                break
            fx = evaluate(point)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = point.copy()

        if best_x is None:
            return None, np.inf

        # Phase 2: Local refinement using Nelder-Mead with strict call limit
        # Remaining budget
        remaining = budget - calls
        
        if remaining > 10 and time.time() < deadline:
            # Use Nelder-Mead which is derivative-free and robust for small dims
            # We wrap func to count calls and enforce limit
            nm_calls = 0
            nm_limit = min(remaining, 200) # Cap NM calls to be safe
            
            def nm_func(x):
                nonlocal nm_calls
                if nm_calls >= nm_limit:
                    return best_f # Return current best to stop improvement
                x = np.clip(x, -5.0, 5.0)
                fx = float(func(x))
                nm_calls += 1
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx

            # Set up Nelder-Mead
            # Initial simplex size can be small since we are in a promising basin
            # Default is fine, but we can tweak if needed.
            # Maxiter is a safety net, but we rely on nm_limit via wrapper?
            # Actually, Nelder-Mead doesn't have a direct max_evals in scipy.optimize.minimize
            # We must break out or let it run. 
            # Better: use a custom loop or just let it run and check time/budget.
            # Since nm_func increments global 'calls', the main loop will stop if budget hits.
            # But minimize() might keep running even if we want to stop.
            # We can pass maxiter to limit iterations.
            # Each iteration uses ~dim+1 evaluations.
            # So maxiter = nm_limit // (dim + 1)
            max_iter = max(1, nm_limit // (dim + 1))
            
            try:
                res = minimize(
                    nm_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                # Update best if NM found something better (already done in nm_func)
            except Exception:
                pass # Ignore errors, keep current best

        return best_x, best_f
# EVOLVE-BLOCK-END
