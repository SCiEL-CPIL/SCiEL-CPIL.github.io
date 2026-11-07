import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        
        # Phase 1: Global Search
        # Allocate 30% of budget for global exploration to find basins.
        n_global = int(budget * 0.3)
        n_global = max(n_global, 30)
        
        # Generate LHS points
        lhs = np.zeros((n_global, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n_global + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points)
            lhs[:, d] = points

        f_values = np.zeros(n_global)
        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)
        
        for i in range(n_global):
            if time.time() > deadline:
                break
            try:
                f_val = func(lhs[i])
                f_values[i] = f_val
                calls += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = lhs[i].copy()
            except Exception:
                break

        if calls == 0:
            return best_x, 0.0

        sorted_indices = np.argsort(f_values)
        
        # Phase 2: Local Refinement
        # Use Nelder-Mead for robustness on multimodal functions, with adaptive restarts.
        # Powell can sometimes get stuck in local minima or fail to converge in high dimensions
        # if the initial simplex is poor. NM is generally more robust for black-box functions.
        
        n_local_candidates = 5
        remaining_budget = budget - calls
        
        if remaining_budget <= 0:
            return best_x, best_f
        
        # We will distribute the remaining budget among the top candidates.
        # We give more budget to the better candidates.
        # Candidate 0 gets 40% of remaining, Candidate 1 gets 25%, etc.
        # But we also want to ensure we have enough calls for a meaningful optimization.
        
        # Let's try a simpler allocation: equal split, but with a minimum.
        # Actually, let's prioritize the best one heavily.
        # Top 1: 50% of remaining
        # Top 2: 20%
        # Top 3: 10%
        # Top 4: 10%
        # Top 5: 10%
        
        weights = [0.4, 0.2, 0.15, 0.1, 0.15]
        # Normalize weights to sum to 1
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        
        call_counter = [0]
        
        def wrapped_func(x):
            if call_counter[0] >= budget - calls or time.time() > deadline:
                return float('inf')
            val = func(x)
            call_counter[0] += 1
            return val

        for k in range(n_local_candidates):
            if calls >= budget or time.time() > deadline:
                break
            
            x0 = lhs[sorted_indices[k]].copy()
            f0 = f_values[sorted_indices[k]]
            
            # Calculate budget for this restart
            budget_for_restart = int(remaining_budget * weights[k])
            if budget_for_restart < 10:
                budget_for_restart = 10
            
            try:
                # Nelder-Mead is robust
                result = minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                # Check if result is valid
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
            
            calls += call_counter[0]
            call_counter[0] = 0

        return best_x, best_f
