"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start Powell optimizer for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_val = np.inf
        best_x = np.zeros(dim)

        def call_func(x):
            nonlocal calls_used, best_val, best_x
            if calls_used >= budget:
                return best_val
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = float(func(x))
            calls_used += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
            return val

        # Phase 1: Random Sampling to find good starting points
        # Use 10% of budget for exploration
        n_init = max(10, budget // 10)
        points = []
        values = []
        
        # Always evaluate center
        f0 = call_func(np.zeros(dim))
        points.append(np.zeros(dim))
        values.append(f0)

        rng = np.random.default_rng(42)
        for _ in range(n_init - 1):
            if calls_used >= budget or time.time() > deadline - 60:
                break
            x = rng.uniform(-5, 5, dim)
            f = call_func(x)
            points.append(x)
            values.append(f)

        points = np.array(points)
        values = np.array(values)
        
        # Get top 3 starting points
        top_indices = np.argsort(values)[:3]
        start_points = points[top_indices]

        # Phase 2: Local Optimization with Powell
        # Distribute remaining budget among the starts
        remaining_budget = budget - calls_used
        if remaining_budget > 0:
            # Allocate budget per start
            budget_per_start = max(10, remaining_budget // len(start_points))
            
            for i, x_start in enumerate(start_points):
                if calls_used >= budget or time.time() > deadline - 30:
                    break
                
                # Limit the number of function evaluations for this specific minimize call
                # Powell can be verbose in calls, so we cap maxfev
                maxfev = min(budget_per_start, budget - calls_used)
                
                try:
                    res = minimize(
                        call_func,
                        x_start,
                        method='Powell',
                        options={'maxiter': 50, 'maxfev': maxfev, 'xtol': 1e-4, 'ftol': 1e-4}
                    )
                except Exception:
                    pass

        return best_x, best_val
# EVOLVE-BLOCK-END
