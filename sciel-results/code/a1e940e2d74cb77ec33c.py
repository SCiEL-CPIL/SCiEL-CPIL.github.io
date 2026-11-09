"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB f17 (Schaffer's F7)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 330s limit, leave 40s margin

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            fx = float(func(x))
            calls_made += 1
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        # Phase 1: Global Sampling
        # Use a mix of random and structured points to ensure coverage
        n_global = 40
        points = []
        
        # Center
        points.append(np.zeros(d))
        
        # Corners and axis extremes
        for corner in [(1,1), (-1,1), (1,-1), (-1,-1)]:
            pts = []
            for i in range(d):
                if i % 2 == 0:
                    pts.append(corner[0] * 4.0)
                else:
                    pts.append(corner[1] * 4.0)
            points.append(np.array(pts))
            
        for i in range(d):
            x = np.zeros(d)
            x[i] = 4.0
            points.append(x.copy())
            x[i] = -4.0
            points.append(x.copy())

        # Random points
        while len(points) < n_global:
            points.append(rng.uniform(-4.5, 4.5, d))

        # Evaluate global points
        fvals = []
        for x in points:
            if not time_ok():
                break
            x = np.clip(x, -5, 5)
            fx = call_func(x)
            fvals.append(fx)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)

        # Sort points by fitness
        indices = np.argsort(fvals)
        top_points = [points[i] for i in indices[:min(5, len(indices))]]

        # Phase 2: Local Refinement with Nelder-Mead
        # Distribute remaining budget among top points
        remaining_budget = budget - calls_made
        if remaining_budget > 0:
            n_refine = len(top_points)
            budget_per_point = max(10, remaining_budget // n_refine)
            
            for x0 in top_points:
                if not time_ok():
                    break
                if calls_made >= budget:
                    break
                
                # Define a wrapper that stops if budget is exceeded
                # Nelder-Mead uses 'maxiter' but also stops on tolerance
                # We need to be careful with the call counting
                def objective(x):
                    if calls_made >= budget:
                        return np.inf
                    return call_func(np.clip(x, -5, 5))
                
                try:
                    res = minimize(objective, x0, method='Nelder-Mead', 
                                   options={'maxiter': budget_per_point, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
                    if res.fun < best_fx:
                        best_fx = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Phase 3: Final Polish
        if calls_made < budget and time.time() < deadline:
            # A few random steps around the best point
            for _ in range(min(10, budget - calls_made)):
                if not time_ok():
                    break
                y = best_x + 0.01 * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                fy = call_func(y)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
