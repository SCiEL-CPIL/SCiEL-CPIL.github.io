import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 285.0

        budget = self.budget
        d = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline or calls >= budget:
                raise StopIteration()
            f = float(func(np.asarray(x, dtype=np.float64)))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=np.float64).copy()
            return f

        # Phase 1: Multi-start exploration
        n_starts = 3
        explore_budget = int(budget * 0.4)
        per_start_budget = explore_budget // n_starts
        
        for start_idx in range(n_starts):
            if calls >= explore_budget or time.time() > deadline:
                break
            
            if start_idx == 0:
                x = np.zeros(d)
            else:
                x = rng.uniform(-4, 4, d)
            
            try:
                f_x = safe_func(x)
            except StopIteration:
                break
            
            sigma = 1.5
            stall_count = 0
            stall_limit = 30
            growth = 1.15
            decay = 0.85
            min_sigma = 0.05
            max_sigma = 4.0
            
            local_calls = 0
            while calls < explore_budget and local_calls < per_start_budget and time.time() < deadline:
                step = rng.normal(0, 1, d)
                x_candidate = x + sigma * step
                
                try:
                    f_candidate = safe_func(x_candidate)
                except StopIteration:
                    break
                
                local_calls += 1
                
                if f_candidate < f_x:
                    x = x_candidate
                    f_x = f_candidate
                    sigma = min(sigma * growth, max_sigma)
                    stall_count = 0
                else:
                    sigma *= decay
                    stall_count += 1
                    
                if stall_count >= stall_limit:
                    if sigma > min_sigma:
                        sigma = max(sigma * 0.5, min_sigma)
                    else:
                        break
                    stall_count = 0

        # Phase 2: Intensive local refinement from best point
        if best_x is not None and calls < budget:
            x = best_x.copy()
            f_x = best_f
            
            # Use coordinate descent + (1+1)-ES hybrid
            refine_budget = budget - calls
            sigma = 0.2
            coord_sigma = 0.1
            
            for i in range(min(refine_budget, 200)):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Try coordinate moves
                improved = False
                for dim in range(d):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    for direction in [1, -1]:
                        x_trial = x.copy()
                        x_trial[dim] += direction * coord_sigma
                        
                        try:
                            f_trial = safe_func(x_trial)
                        except StopIteration:
                            break
                        
                        if f_trial < f_x:
                            x = x_trial
                            f_x = f_trial
                            improved = True
                            break
                
                if improved:
                    coord_sigma *= 0.9
                    sigma *= 1.05
                else:
                    # Try (1+1)-ES step
                    step = rng.normal(0, 1, d)
                    x_candidate = x + sigma * step
                    
                    try:
                        f_candidate = safe_func(x_candidate)
                    except StopIteration:
                        break
                    
                    if f_candidate < f_x:
                        x = x_candidate
                        f_x = f_candidate
                        sigma *= 1.1
                        coord_sigma *= 1.05
                    else:
                        sigma *= 0.9
                        coord_sigma *= 0.9
                        
                    if sigma < 1e-5 and coord_sigma < 1e-5:
                        break

        return best_x, best_f
