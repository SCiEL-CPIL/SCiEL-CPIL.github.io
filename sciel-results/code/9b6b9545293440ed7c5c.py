"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid-based global search followed by coordinate descent with Nelder-Mead-like steps."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270  # 30s safety margin before 300s limit

        calls = 0

        def evaluate(x):
            nonlocal calls
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            return x, fx

        # Phase 1: Coarse grid search to find a good basin
        # Use 3 points per dimension: -4, 0, 4 (within [-4,4] where minimizer lies)
        # 3^5 = 243 evaluations
        grid_points = []
        for d in range(n):
            grid_points.append([-4.0, 0.0, 4.0])
        
        # Generate all 3^5 combinations
        import itertools
        best_x = None
        best_f = np.inf
        
        for combo in itertools.product(*grid_points):
            if time.time() > deadline:
                break
            x = np.array(combo, dtype=float)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback: random point
            best_x = rng.uniform(-5, 5, n)
            best_f = float(func(best_x))
            calls += 1

        # Phase 2: Fine grid around the best point
        # Use 3 points per dimension in a smaller neighborhood
        center = best_x.copy()
        step = 2.0  # neighborhood half-width
        
        for d in range(n):
            if time.time() > deadline:
                break
            # Only refine dimensions where we can afford it
            # Use 3 points: center - step, center, center + step
            lo = max(-5.0, center[d] - step)
            hi = min(5.0, center[d] + step)
            if lo >= hi:
                continue
            mid = (lo + hi) / 2.0
            offsets = [lo, mid, hi]
            
            # Evaluate variations along this dimension
            for offset in offsets:
                if time.time() > deadline or calls >= budget:
                    break
                x_new = center.copy()
                x_new[d] = offset
                x_new = np.clip(x_new, -5.0, 5.0)
                fx_new = float(func(x_new))
                calls += 1
                if fx_new < best_f:
                    best_f = fx_new
                    center = x_new.copy()
        
        # Phase 3: Coordinate descent with adaptive step size
        # Use remaining budget for iterative refinement
        step_size = 1.0
        no_improve_count = 0
        max_no_improve = 20
        
        while calls < budget and time.time() < deadline:
            improved = False
            
            for d in range(n):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Try positive and negative steps
                for direction in [1.0, -1.0]:
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    x_try = best_x.copy()
                    x_try[d] += direction * step_size
                    x_try = np.clip(x_try, -5.0, 5.0)
                    
                    fx_try = float(func(x_try))
                    calls += 1
                    
                    if fx_try < best_f:
                        best_x = x_try.copy()
                        best_f = fx_try
                        improved = True
                        step_size = max(step_size * 1.2, 0.01)
                        no_improve_count = 0
                    else:
                        step_size = max(step_size * 0.7, 0.001)
            
            if not improved:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    # Global random restart
                    if time.time() > deadline:
                        break
                    x_restart = rng.uniform(-5, 5, n)
                    fx_restart = float(func(x_restart))
                    calls += 1
                    if fx_restart < best_f:
                        best_x = x_restart.copy()
                        best_f = fx_restart
                        no_improve_count = 0
                        step_size = 2.0
                    else:
                        # Keep current best, reset step
                        step_size = 1.0
                        no_improve_count = 0
                        # If we've tried many restarts without improvement, break
                        if no_improve_count > 5:
                            break
            else:
                no_improve_count = 0
        
        return best_x, best_f
# EVOLVE-BLOCK-END
