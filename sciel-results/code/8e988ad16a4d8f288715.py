"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead with diversified sampling and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        dim = self.dim
        total_budget = self.budget
        bounds = [(-5.0, 5.0) for _ in range(dim)]

        def safe_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Diversified Initialization
        # Sample enough points to likely hit the global basin of f6
        n_initial = min(30, total_budget // 3)
        if n_initial < 10:
            n_initial = 10
            
        initial_points = []
        # Include center
        initial_points.append(np.zeros(dim))
        # Include some structured points
        for i in range(dim):
            x = np.zeros(dim)
            x[i] = 2.0
            initial_points.append(x.copy())
            x[i] = -2.0
            initial_points.append(x.copy())
        
        # Fill remaining with random points
        while len(initial_points) < n_initial:
            x0 = rng.uniform(-5, 5, dim)
            initial_points.append(x0)

        init_results = []
        for x0 in initial_points[:n_initial]:
            if calls_made >= total_budget or time.time() > deadline:
                break
            fx = safe_func(x0)
            if np.isfinite(fx):
                init_results.append((fx, x0.copy()))

        if not init_results:
            x0 = np.zeros(dim)
            fx = safe_func(x0)
            return best_x, best_f

        # Sort initial results
        init_results.sort(key=lambda t: t[0])
        
        # Phase 2: Local Optimization using Nelder-Mead from the best starts
        # Nelder-Mead is more robust for 5D local search than L-BFGS-B
        k_local = min(8, len(init_results))
        for i in range(k_local):
            if calls_made >= total_budget or time.time() > deadline:
                break
            
            fx_init, x_init = init_results[i]
            
            # Run Nelder-Mead local optimization
            try:
                res = minimize(safe_func, x_init, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4,
                                        'maxfev': min(50, total_budget - calls_made)})
            except Exception:
                pass
            
            # Check if we should continue
            if time.time() > deadline:
                break

        # Phase 3: If we have budget left, do additional random restarts + local search
        while calls_made < total_budget and time.time() < deadline:
            # Generate a new random point
            x0 = rng.uniform(-5, 5, dim)
            fx0 = safe_func(x0)
            
            if np.isinf(fx0):
                break
                
            # If this point is reasonably good, run a short local optimization
            if calls_made + 20 < total_budget:
                try:
                    res = minimize(safe_func, x0, method='Nelder-Mead',
                                   options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-3,
                                            'maxfev': min(20, total_budget - calls_made)})
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
