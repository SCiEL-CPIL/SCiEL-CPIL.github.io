"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid Scout + Powell + Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        calls = 0
        self.budget = budget  # Ensure local access

        def wrapped_func(x):
            nonlocal calls
            if calls >= self.budget:
                return 1e10
            if time.time() > deadline:
                return 1e10
            calls += 1
            return func(x)

        # 1. Scout Phase: Evaluate multiple strategic points
        # Rosenbrock base minimum is at (1,1,1,1,1).
        # Shifts/rotations might move it, but usually not far.
        candidates = [
            np.zeros(self.dim),
            np.ones(self.dim),
            -np.ones(self.dim),
            np.full(self.dim, 2.0),
            np.full(self.dim, -2.0)
        ]
        
        # Add a few random points to catch larger shifts
        rng = np.random.default_rng(42)
        for _ in range(3):
            candidates.append(rng.uniform(-4, 4, self.dim))

        best_x = None
        best_f = float('inf')
        
        for x0 in candidates:
            # Clip to box just in case, though func does it
            x0 = np.clip(x0, -5, 5)
            fx = wrapped_func(x0)
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()
        
        # 2. Local Optimization Phase 1: Powell
        # Powell is efficient for Rosenbrock.
        # Estimate maxiter based on remaining budget.
        # Heuristic: ~30-50 function calls per Powell iteration in 5D.
        remaining_calls = self.budget - calls
        max_iter_powell = max(1, remaining_calls // 40)
        
        try:
            res = minimize(wrapped_func, best_x, method='Powell', 
                           options={'maxiter': max_iter_powell, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass

        # 3. Local Optimization Phase 2: Nelder-Mead for fine polishing
        remaining_calls = self.budget - calls
        if remaining_calls > 10:
            # Nelder-Mead uses ~1 call per iteration (reflection)
            max_iter_nm = max(1, remaining_calls // 2) # Conservative
            try:
                res2 = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                                options={'maxiter': max_iter_nm, 'xatol': 1e-8, 'fatol': 1e-10})
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
