import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)
        
        lb = np.full(self.dim, -5.0)
        ub = np.full(self.dim, 5.0)
        bounds = [(-5.0, 5.0)] * self.dim

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                return np.inf
            
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Diverse Global Sampling
        # We want to cover the space well.
        # 1. Uniform random
        # 2. Axis-aligned points (to probe individual dimensions)
        # 3. Points near the origin (common in BBOB)
        
        candidates = []
        
        # 1. Uniform random points
        n_uniform = 15
        for _ in range(n_uniform):
            x = rng.uniform(-5.0, 5.0, self.dim)
            candidates.append(x)
            
        # 2. Axis-aligned points
        # For each dimension, try +/- 4.0 (near boundary) and 0.0
        for d in range(self.dim):
            for val in [-4.0, 0.0, 4.0]:
                x = np.zeros(self.dim)
                x[d] = val
                candidates.append(x)
                
        # 3. Origin and small perturbations
        candidates.append(np.zeros(self.dim))
        for _ in range(5):
            x = rng.normal(0.0, 0.5, self.dim)
            candidates.append(x)

        # Evaluate all candidates
        evaluated_points = []
        for x in candidates:
            if time.time() > deadline or calls_made >= self.budget:
                break
            f = safe_func(x)
            evaluated_points.append((f, x))
            
        # Sort by fitness
        evaluated_points.sort(key=lambda item: item[0])
        
        # Phase 2: Local Optimization from Top K
        # Pick top K candidates to refine
        K = 5
        top_candidates = [x for f, x in evaluated_points[:K]]
        
        # Distribute remaining budget among top candidates
        remaining_budget = self.budget - calls_made
        if remaining_budget > 0 and top_candidates:
            # Give each top candidate a share of the budget
            # L-BFGS-B is efficient, so we can afford to give them decent budgets
            budget_per_candidate = max(10, remaining_budget // len(top_candidates))
            
            for x0 in top_candidates:
                if time.time() > deadline or calls_made >= self.budget:
                    break
                    
                # Cap the maxfev for this specific optimization run
                max_fev = min(budget_per_candidate, self.budget - calls_made)
                
                try:
                    res = minimize(
                        safe_func,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxfev': max_fev,
                            'ftol': 1e-14,
                            'gtol': 1e-10,
                            'maxiter': 100
                        }
                    )
                except Exception:
                    pass

        # Final check and return
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            
        return best_x, best_f
