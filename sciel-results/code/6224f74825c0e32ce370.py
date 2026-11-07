import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin for 300s limit
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(12345)

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            try:
                f = func(x)
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                return np.inf

        # Phase 1: Global Exploration
        # Initialize 50 random points
        n_initial = 50
        initial_points = []
        for _ in range(n_initial):
            if calls_made >= budget or time.time() > deadline:
                break
            p = rng.uniform(-5, 5, dim)
            f = safe_call(p)
            if f < np.inf:
                initial_points.append((f, p))
        
        if not initial_points:
            # Fallback
            p = np.zeros(dim)
            f = safe_call(p)
            if f < np.inf:
                initial_points.append((f, p))
        
        if not initial_points:
            return best_x, best_f

        # Sort by fitness and take top 5
        initial_points.sort(key=lambda x: x[0])
        top_k = min(5, len(initial_points))
        # starts is a list of tuples (f, x)
        starts = initial_points[:top_k]

        # Phase 2: Parallel (1+1)-ES from each top start
        # Allocate remaining budget evenly
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f
        
        budget_per_es = max(1, remaining_budget // top_k)
        
        target_sr = 0.2 # Target success rate for step size adaptation
        
        for f_start, x_start in starts:
            if calls_made >= budget or time.time() > deadline:
                break
                
            x = x_start.copy()
            f = f_start
            step_size = 1.0
            success_count = 0
            total_steps = 0
            local_steps = 0
            
            # Run (1+1)-ES for this lineage
            while local_steps < budget_per_es and calls_made < budget and time.time() < deadline:
                # Mutation
                mutation = rng.normal(0, step_size, dim)
                x_new = x + mutation
                x_new = np.clip(x_new, -5, 5)
                
                f_new = safe_call(x_new)
                local_steps += 1
                
                if f_new < f:
                    # Success
                    x = x_new
                    f = f_new
                    success_count += 1
                    total_steps += 1
                else:
                    total_steps += 1
                
                # Adapt step size every 5 steps
                if total_steps > 0 and total_steps % 5 == 0:
                    sr = success_count / total_steps
                    if sr > target_sr:
                        step_size *= 1.2
                    elif sr < target_sr:
                        step_size /= 1.2
                    # Reset counters
                    success_count = 0
                    total_steps = 0

        return best_x, best_f
