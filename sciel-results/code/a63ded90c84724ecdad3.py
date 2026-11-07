import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # 20s margin
        
        # Phase 1: Global Sampling
        n_samples = 10
        if budget < n_samples:
            # Fallback to simple random search
            # ...
            pass
        
        points = np.random.uniform(-5, 5, (n_samples, dim))
        best_x = None
        best_f = np.inf
        
        eval_count = 0
        
        for p in points:
            if time.time() > deadline:
                break
            f_val = func(p)
            eval_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = p.copy()
                
        # Phase 2: Local Refinement
        # Use the top 3 candidates
        # We need to store the top 3 from phase 1
        # Let's re-do phase 1 to keep top 3 properly
        
        candidates = []
        for p in points:
            if time.time() > deadline:
                break
            f_val = func(p)
            eval_count += 1
            candidates.append((f_val, p.copy()))
        
        candidates.sort(key=lambda x: x[0])
        top_candidates = candidates[:3]
        
        # Refine each top candidate
        remaining_budget = budget - eval_count
        budget_per_refine = remaining_budget // len(top_candidates) if top_candidates else 0
        
        for f_init, x_init in top_candidates:
            if time.time() > deadline:
                break
            if budget_per_refine <= 0:
                break
            
            # Wrapper to limit evaluations
            local_evals = 0
            local_limit = budget_per_refine
            
            def limited_func(x):
                nonlocal local_evals
                if local_evals >= local_limit:
                    raise StopIteration
                local_evals += 1
                return func(x)
                
            try:
                res = scipy.optimize.minimize(
                    limited_func, 
                    x_init, 
                    method='Powell', 
                    options={'maxiter': 1000, 'maxfev': local_limit}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except StopIteration:
                pass
            except Exception:
                pass
                
        return best_x, best_f
