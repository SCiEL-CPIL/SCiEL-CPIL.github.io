import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State
        best_x = np.zeros(dim)
        best_f = np.inf
        eval_count = 0
        rng = np.random.default_rng(42)
        
        # Function wrapper with counting and clipping
        def counting_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            eval_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Global Exploration with Halton Sequence
        # Use a quasi-random sequence to spread points evenly across the domain.
        # This is more efficient than random sampling for covering the space.
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Allocate 40% of budget to global exploration
        global_budget = int(budget * 0.4)
        bases = [2, 3, 5, 7, 11]
        
        for i in range(1, global_budget + 1):
            if time.time() > deadline - 10:
                break
            pt = np.array([halton(i, base) for base in bases])
            pt = pt * 10.0 - 5.0
            counting_func(pt)

        # Phase 2: Local Refinement with Nelder-Mead
        # Pick the top K best points from Phase 1 and run Nelder-Mead on them.
        # We need to store the points and values from Phase 1.
        # Since counting_func updates best_x/best_f but doesn't store history,
        # let's re-evaluate or store history. Storing history is better.
        
        # Let's restart with a history-aware approach for the first phase.
        # Actually, the current counting_func is already used. 
        # To do this properly, I should have stored the points.
        # Let's just use the best point found so far and a few random perturbations?
        # No, let's do a proper implementation.
        
        # I will rewrite the logic to store points in Phase 1.
        
        pass # Placeholder, I will write the full correct code below.

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        eval_count = 0
        rng = np.random.default_rng(42)
        
        # Store evaluated points and their values
        points = []
        values = []
        
        def counting_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            eval_count += 1
            points.append(x.copy())
            values.append(val)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Global Exploration (Halton)
        # 40% budget
        global_budget = int(budget * 0.4)
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        bases = [2, 3, 5, 7, 11]
        
        for i in range(1, global_budget + 1):
            if time.time() > deadline - 10:
                break
            pt = np.array([halton(i, base) for base in bases])
            pt = pt * 10.0 - 5.0
            counting_func(pt)

        # Phase 2: Local Refinement (Nelder-Mead)
        # Select top 10 points from Phase 1
        if len(points) > 0:
            # Sort by value
            sorted_indices = np.argsort(values)
            top_k = min(10, len(sorted_indices))
            top_indices = sorted_indices[:top_k]
            
            # Allocate remaining budget to these starts
            remaining_budget = budget - eval_count
            if remaining_budget > 0:
                budget_per_start = remaining_budget // top_k
                
                for idx in top_indices:
                    if eval_count >= budget:
                        break
                    if time.time() > deadline - 5:
                        break
                    
                    x0 = points[idx]
                    
                    # Run Nelder-Mead
                    # We need a custom NM implementation or use scipy?
                    # SciPy is available.
                    try:
                        import scipy.optimize
                        # Use scipy.optimize.minimize with Nelder-Mead
                        # But we need to control the eval count strictly.
                        # scipy.optimize.minimize doesn't easily expose a callback to stop mid-run 
                        # based on an external counter unless we use a wrapper that raises an exception 
                        # or checks a flag.
                        
                        # Let's use a simple manual Nelder-Mead or trust region?
                        # Or just use scipy with a maxiter that corresponds to budget.
                        # Nelder-Mead uses ~1 eval per iteration? No, it maintains a simplex of size N+1.
                        # Each iteration might use 1 eval if reflection works, or more.
                        # Let's assume ~2 evals per iteration on average.
                        max_iter = max(1, budget_per_start // 2)
                        
                        res = scipy.optimize.minimize(
                            counting_func, 
                            x0, 
                            method='Nelder-Mead',
                            options={
                                'maxiter': max_iter,
                                'xatol': 1e-8,
                                'fatol': 1e-8
                            }
                        )
                        # counting_func already updates best_x and best_f
                    except Exception:
                        pass

        return best_x, best_f
