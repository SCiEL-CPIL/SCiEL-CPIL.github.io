"""Combined optimizer: Coarse-to-fine 1D grid search + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizes BBOB f3 (Rastrigin) using a deterministic coarse-to-fine 1D grid search
    followed by Nelder-Mead polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill

        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse-to-fine 1D grid search
        # Initialize at origin
        x = np.zeros(dim)
        fx = safe_call(x)

        # Grid sizes: start coarse, get finer
        grid_sizes = [10, 20, 40, 80]

        for gs in grid_sizes:
            if calls_made >= budget or time.time() > deadline:
                break

            improved = True
            while improved and calls_made < budget and time.time() < deadline:
                improved = False
                for i in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    # 1D scan along dimension i
                    # Scan range: [x[i] - 5, x[i] + 5] clipped to [-5, 5]
                    lo = max(-5.0, x[i] - 5.0)
                    hi = min(5.0, x[i] + 5.0)
                    
                    # If the range is too small, skip
                    if hi - lo < 1e-6:
                        continue
                        
                    grid = np.linspace(lo, hi, gs)
                    best_val = fx
                    best_coord = x[i]
                    
                    for g in grid:
                        if calls_made >= budget or time.time() > deadline:
                            break
                        test_x = x.copy()
                        test_x[i] = g
                        tv = safe_call(test_x)
                        if tv < best_val:
                            best_val = tv
                            best_coord = g
                            
                    if best_coord != x[i]:
                        x[i] = best_coord
                        fx = best_val
                        improved = True

        # Phase 2: Nelder-Mead polish from best point found
        if calls_made < budget and time.time() < deadline:
            try:
                remaining = budget - calls_made
                if remaining > 10:
                    max_iter = min(remaining // 2, 200)
                    result = minimize(
                        func,
                        x0=best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': max_iter,
                            'xatol': 1e-10,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    # Check if result is better
                    if result.fun < best_f:
                        best_x = result.x
                        best_f = result.fun
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
