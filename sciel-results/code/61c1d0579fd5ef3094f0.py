import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def clip(x):
            return np.clip(x, -5, 5)

        def remaining_time():
            return deadline - time.time()

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = clip(np.asarray(x, dtype=float))
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start with diverse initial points
        # For Katsuura, multimodality is key. Use structured + random starts.
        initial_points = []
        
        # Center
        initial_points.append(np.zeros(dim))
        
        # Vertices of [-5, 5]^dim (sample a subset to save budget)
        for _ in range(8):
            initial_points.append(rng.uniform(-5, 5, dim))
        
        # Some points near the known range of the minimum [-4, 4]^5
        for _ in range(8):
            initial_points.append(rng.uniform(-4, 4, dim))
        
        # Evaluate all initial points
        for x0 in initial_points:
            if calls_made >= budget or remaining_time() < 10.0:
                break
            x0 = clip(x0)
            evaluate(x0)
        
        if best_x is None:
            # Should not happen, but safety
            best_x = np.zeros(dim)
            best_f = float('inf')
            evaluate(best_x)

        # Phase 2: Local refinement using Nelder-Mead from multiple top candidates
        # We don't have a population, so we'll just refine the best and a few nearby random perturbations
        # But to be safe and use the budget, let's do a few (1+1)-ES runs from promising starts
        
        # Actually, let's use scipy's minimize with Nelder-Mead from the best point
        # and a few random perturbations of the best point.
        
        # First, refine the current best
        if calls_made < budget and remaining_time() > 10.0:
            # Use a wrapper to track calls and enforce budget
            local_calls = [0]
            local_budget = min(budget - calls_made, 200)  # Cap local refinement to 200 calls per start
            
            def local_func(x):
                local_calls[0] += 1
                if local_calls[0] > local_budget:
                    return best_f + 1.0  # Penalize to stop
                return evaluate(x)
            
            try:
                res = minimize(local_func, best_x, method='Nelder-Mead', 
                             options={'maxiter': local_budget, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = clip(res.x)
            except Exception:
                pass

        # Phase 3: More global exploration with (1+1)-ES from diverse points
        # Use remaining budget for (1+1)-ES restarts
        remaining_calls = budget - calls_made
        if remaining_calls > 50 and remaining_time() > 10.0:
            # Run several (1+1)-ES attempts
            num_es_attempts = min(10, remaining_calls // 30)
            for i in range(num_es_attempts):
                if calls_made >= budget or remaining_time() < 5.0:
                    break
                
                # Start from best + perturbation or random
                if i == 0:
                    x = clip(best_x + 1.0 * rng.standard_normal(dim))
                else:
                    if rng.random() < 0.5:
                        x = clip(best_x + 2.0 * rng.standard_normal(dim))
                    else:
                        x = clip(rng.uniform(-5, 5, dim))
                
                fx = evaluate(x)
                sigma = 0.5
                no_improve = 0
                max_steps = 30
                
                for _ in range(max_steps):
                    if calls_made >= budget or remaining_time() < 2.0:
                        break
                    
                    y = clip(x + sigma * rng.standard_normal(dim))
                    fy = evaluate(y)
                    
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.1
                        no_improve = 0
                    else:
                        sigma *= 0.9
                        no_improve += 1
                    
                    if no_improve >= 15:
                        break
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 4: Final fine-tuning with very small steps
        if calls_made < budget and remaining_time() > 5.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            
            while calls_made < budget and remaining_time() > 1.0:
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.01
                else:
                    sigma *= 0.95
                    if sigma < 1e-5:
                        break

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
