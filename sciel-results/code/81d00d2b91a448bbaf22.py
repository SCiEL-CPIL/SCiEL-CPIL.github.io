import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin

        calls_made = 0
        best_f = float('inf')
        best_x = None

        def safe_func(x):
            nonlocal calls_made, best_f, best_x
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)

        # Phase 1: Global Sampling
        # Use 20% of budget for initial sampling to find a good basin
        n_init = max(10, total_budget // 5)
        init_points = []
        
        # Ensure diversity: mix random and structured
        # 1. Center
        init_points.append(np.zeros(dim))
        # 2. Random points in [-4, 4]^5
        for _ in range(n_init - 1):
            pt = rng.uniform(-4.0, 4.0, dim)
            init_points.append(pt)

        init_vals = []
        for pt in init_points:
            if time.time() > deadline or calls_made >= total_budget:
                break
            val = safe_func(pt)
            init_vals.append((val, pt))
        
        if not init_vals:
            # Fallback if no calls made
            return np.zeros(dim), float('inf')

        # Sort by value
        init_vals.sort(key=lambda x: x[0])
        
        # Phase 2: Local Refinement with Nelder-Mead
        # Pick the top 3 candidates for local search
        # Allocate remaining budget to them
        remaining_budget = total_budget - calls_made
        n_local_starts = min(3, len(init_vals))
        
        if n_local_starts > 0 and remaining_budget > 0:
            # Divide budget among local searches
            # Nelder-Mead in 5D uses ~6 calls/iter. 
            # We want to use the budget effectively.
            budget_per_start = remaining_budget // n_local_starts
            
            for i in range(n_local_starts):
                if time.time() > deadline or calls_made >= total_budget:
                    break
                
                start_x = init_vals[i][1].copy()
                
                # If the start point is already very good, use a smaller initial step?
                # Nelder-Mead default initial step is 0.05 * x0 or 0.00025 if x0 is 0.
                # For Weierstrass, the basin is not extremely narrow, so default is okay.
                
                try:
                    # Use maxiter to control duration roughly
                    # Empirically, Nelder-Mead in 5D uses ~5-10 calls per iteration
                    # So maxiter = budget_per_start / 5 is a safe upper bound
                    max_iter = max(10, budget_per_start // 5)
                    
                    result = minimize(
                        safe_func,
                        start_x,
                        method='Nelder-Mead',
                        options={'maxiter': max_iter,
                                 'xatol': 1e-6, 
                                 'fatol': 1e-8}
                    )
                except Exception:
                    pass

        # Phase 3: Final Polish if budget remains
        # If we have calls left, do a few random perturbations around best_x
        if calls_made < total_budget and best_x is not None:
            while calls_made < total_budget and time.time() < deadline:
                # Small random step
                step = 0.1 * rng.standard_normal(dim)
                new_x = np.clip(best_x + step, -5.0, 5.0)
                safe_func(new_x)
                # If we found something better, continue, else break if no improvement in last 10?
                # Simple loop for now

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
