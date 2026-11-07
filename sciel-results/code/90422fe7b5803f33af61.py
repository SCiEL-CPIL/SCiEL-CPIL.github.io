import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 290.0
        
        budget = self.budget
        dim = self.dim
        calls = 0
        best_f = float('inf')
        best_x = None
        
        # Helper to evaluate and track
        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            val = func(x)
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val
        
        # Phase 1: Global exploration
        # Sample a moderate number of points to find the basin of attraction for the sharp ridge
        # 30 points is a good balance between exploration and leaving budget for exploitation
        n_explore = min(30, budget)
        explore_points = []
        explore_vals = []
        
        for _ in range(n_explore):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = evaluate(x)
            if f is not None:
                explore_points.append(x)
                explore_vals.append(f)
        
        if best_x is None:
            return best_x, best_f
        
        # Sort exploration points by fitness (best first)
        indices = np.argsort(explore_vals)
        explore_points = [explore_points[i] for i in indices]
        explore_vals = [explore_vals[i] for i in indices]
        
        # Phase 2: Multi-start (1+1)-ES
        # Use the top few points as starting points for ES runs
        # The (1+1)-ES with sigma adaptation (1.5 on success, 0.87 on failure) is proven to work well
        # We'll run ES from the best point first, then from the next best if budget allows
        
        def run_es(x_start, n_evals):
            nonlocal calls, best_f, best_x
            x = x_start.copy()
            f = evaluate(x)
            if f is None:
                return x, best_f
            
            # Per-dimension step sizes
            sigma = np.ones(dim) * 2.0
            
            for i in range(n_evals):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Generate mutant
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5, 5)
                
                f_trial = evaluate(x_trial)
                if f_trial is None:
                    break
                
                # Update success counter and adapt sigma
                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    sigma *= 1.5
                else:
                    sigma *= 0.87
                
                # Clip sigma to reasonable range
                sigma = np.clip(sigma, 1e-8, 10.0)
            
            return x, f
        
        # Run ES from the best exploration point
        remaining = budget - calls
        if remaining > 0:
            # Use most of the remaining budget for the first ES run
            n_es = remaining
            x, f = run_es(explore_points[0], n_es)
        
        # If there's still budget, run ES from the second best point
        remaining = budget - calls
        if remaining > 10 and len(explore_points) > 1:
            n_es = remaining
            x, f = run_es(explore_points[1], n_es)
        
        # If there's still budget, run ES from the third best point
        remaining = budget - calls
        if remaining > 10 and len(explore_points) > 2:
            n_es = remaining
            x, f = run_es(explore_points[2], n_es)
        
        return best_x, best_f
