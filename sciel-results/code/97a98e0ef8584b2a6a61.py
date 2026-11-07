"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """Multi-start Nelder-Mead for Discus (f11)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        call_count = [0]
        
        def wrapped(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return best_f if np.isfinite(best_f) else 0.0
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            if np.isfinite(val) and val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Global Random Search
        # 50 points to find good basins
        n_global = 50
        points = []
        for _ in range(n_global):
            if time.time() > deadline or call_count[0] >= budget:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            f = wrapped(x)
            points.append((f, x))
            
        points.sort(key=lambda p: p[0])
        
        # Phase 2: Local Refinement with Nelder-Mead
        # Top 5 candidates
        n_local = 5
        remaining = budget - call_count[0]
        budget_per_local = max(10, remaining // n_local)
        
        for f, x in points[:n_local]:
            if time.time() > deadline or call_count[0] >= budget:
                break
            try:
                res = minimize(
                    wrapped,
                    x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': budget_per_local,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

        # Phase 3: Use remaining budget for additional random restarts
        if call_count[0] < budget and time.time() < deadline:
            # Estimate remaining budget
            rem = budget - call_count[0]
            # Do as many random restarts with small local polish as possible
            # Each restart: 1 random + 20 local = 21 calls
            n_extra = rem // 21
            for _ in range(n_extra):
                if time.time() > deadline or call_count[0] >= budget:
                    break
                x = rng.uniform(-5.0, 5.0, dim)
                f = wrapped(x)
                try:
                    res = minimize(
                        wrapped,
                        x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 50,
                            'maxfev': 20,
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
