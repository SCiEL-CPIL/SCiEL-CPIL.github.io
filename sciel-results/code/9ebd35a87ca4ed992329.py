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
        deadline = start_time + 280.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        # --- Phase 1: Global Search ---
        # BBOB-21 (Gallagher 101 peaks) has many local optima.
        # We need a good initial spread to find the basin of the global optimum.
        # Use 30% of the budget for global search to allow more time for local refinement.
        
        n_global = int(budget * 0.3)
        n_global = max(n_global, 30)
        
        # Generate LHS points in [-5, 5]^dim
        lhs = np.zeros((n_global, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n_global + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points)
            lhs[:, d] = points

        # Evaluate global points
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

        # Sort by fitness to get top candidates
        sorted_indices = np.argsort(f_values)
        
        # --- Phase 2: Local Refinement ---
        # Use a hybrid approach:
        # 1. Run a derivative-free local optimizer (Powell) on the best candidate to quickly converge.
        # 2. Use the remaining budget for a (1+1)-ES style stochastic local search to escape shallow local minima
        #    or fine-tune if Powell got stuck or if there are multiple close local minima.
        
        remaining_budget = budget - calls
        
        if remaining_budget <= 0:
            return best_x, best_f

        # Allocate 50% of remaining to Powell, 50% to (1+1)-ES
        # Powell is very efficient for smooth valleys, but (1+1)-ES can be more robust to noise or ruggedness.
        # Actually, for BBOB-21, Powell is usually strong. Let's give Powell a decent chunk,
        # but also run (1+1)-ES on the best point found by Powell (or the initial best if Powell fails)
        # to ensure we don't miss a nearby improvement that Powell might step over or converge prematurely to.
        
        # Strategy:
        # 1. Try Powell on the top 3 candidates.
        # 2. Then take the current best point and run a short (1+1)-ES to polish.
        
        n_powell_candidates = 3
        powell_budget = max(10, remaining_budget // 2)
        es_budget = remaining_budget - powell_budget * n_powell_candidates
        if es_budget < 0:
            es_budget = remaining_budget - powell_budget * 1
            n_powell_candidates = 1

        # --- Powell Phase ---
        for k in range(min(n_powell_candidates, len(sorted_indices))):
            if calls >= budget or time.time() > deadline:
                break
            
            x0 = lhs[sorted_indices[k]].copy()
            
            local_calls = 0
            max_local_calls = powell_budget
            
            def local_func(x):
                nonlocal calls, local_calls, best_f, best_x
                if calls >= budget or local_calls >= max_local_calls or time.time() > deadline:
                    raise TimeoutError("Budget exceeded")
                calls += 1
                local_calls += 1
                val = func(x)
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val

            try:
                # Use Powell method
                # maxiter controls the number of passes over the directions.
                # We limit it to ensure we don't loop forever if it doesn't converge.
                res = minimize(local_func, x0, method='Powell', options={'maxiter': 50})
            except TimeoutError:
                pass
            except Exception:
                pass

        # --- (1+1)-ES Phase ---
        # Start from the current best_x
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            f = best_f
            
            # Adaptive step size
            # Start with a small step since we are likely near a local minimum
            sigma = 0.1 
            
            # 1/5th rule parameters
            window_size = 10
            successes = 0
            window = []
            
            # Run until budget or time is up
            while calls < budget and time.time() < deadline:
                # Generate mutant
                mutation = rng.normal(0, sigma, size=dim)
                x_new = x + mutation
                x_new = np.clip(x_new, -5, 5)
                
                # Evaluate mutant
                try:
                    f_new = func(x_new)
                except Exception:
                    break
                calls += 1
                
                # Update window for 1/5th rule
                if f_new < f:
                    successes += 1
                    window.append(1)
                    # Accept
                    x = x_new
                    f = f_new
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    window.append(0)
                
                if len(window) >= window_size:
                    success_rate = np.mean(window[-window_size:])
                    if success_rate > 0.2:
                        sigma *= 1.1
                    elif success_rate < 0.2:
                        sigma *= 0.9
                    window = window[-window_size:]

        return best_x, best_f
