"""Black-box optimizer for BBOB f11 (Discus) on [-5,5]^5 with 500 evaluations."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start L-BFGS-B optimizer with diverse sampling for global coverage."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Track call count
        call_count = [0]
        best_f = np.inf
        best_x = None
        
        def wrapped_func(x):
            if call_count[0] >= budget:
                return best_f if np.isfinite(best_f) else 0.0
            call_count[0] += 1
            try:
                val = func(x)
                if np.isfinite(val) and val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val
            except Exception:
                return 1e10
        
        # Phase 1: Generate diverse starting points
        rng = np.random.default_rng(42)
        
        # Create stratified sampling points
        n_stratified = 15
        stratified_points = np.zeros((n_stratified, dim))
        for i in range(n_stratified):
            for d in range(dim):
                # Stratified: divide [0,1] into n_stratified intervals
                interval_start = i / n_stratified
                interval_end = (i + 1) / n_stratified
                stratified_points[i, d] = -5 + 10 * (interval_start + rng.random() * (interval_end - interval_start))
        
        # Add center and random points
        center = np.zeros(dim)
        random_points = rng.uniform(-4.5, 4.5, size=(10, dim))
        
        # Combine all starting points
        all_starts = np.vstack([stratified_points, [center], random_points])
        all_starts = np.clip(all_starts, -5, 5)
        
        # Evaluate all starting points to find good ones
        start_values = []
        for i, x0 in enumerate(all_starts):
            if call_count[0] >= budget or time.time() > deadline:
                break
            fx0 = wrapped_func(x0)
            start_values.append((fx0, x0))
        
        # Sort by function value (best first)
        start_values.sort(key=lambda x: x[0])
        
        # Phase 2: Local refinement with L-BFGS-B from top starting points
        # Use more iterations for the best starts
        max_local_calls = max(20, (budget - call_count[0]) // max(1, len(start_values)))
        
        for rank, (fx0, x0) in enumerate(start_values):
            if call_count[0] >= budget - 5 or time.time() > deadline:
                break
            
            # Give more budget to better starting points
            if rank < 5:
                maxiter = 80
                maxls = 30
            elif rank < 10:
                maxiter = 50
                maxls = 20
            else:
                maxiter = 30
                maxls = 10
            
            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': maxiter, 'maxls': maxls, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                continue
            
            if time.time() > deadline:
                break
        
        # Phase 3: If we have budget left, do a final polish from the best point
        if best_x is not None and call_count[0] < budget - 10:
            if time.time() < deadline:
                try:
                    result = minimize(
                        wrapped_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 100, 'maxls': 50, 'ftol': 1e-15, 'gtol': 1e-10}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass
        
        # Phase 4: If still have budget, try a few more random restarts
        remaining_budget = budget - call_count[0]
        if remaining_budget > 20 and time.time() < deadline:
            n_extra = min(5, remaining_budget // 15)
            for _ in range(n_extra):
                if call_count[0] >= budget - 5 or time.time() > deadline:
                    break
                x_try = rng.uniform(-4, 4, size=dim)
                try:
                    result = minimize(
                        wrapped_func,
                        x_try,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 30, 'maxls': 10}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    continue
        
        return best_x, best_f
# EVOLVE-BLOCK-END
