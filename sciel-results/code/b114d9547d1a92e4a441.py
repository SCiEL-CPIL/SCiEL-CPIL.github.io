import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 310.0  # Safety margin before 330s kill

        def _time_ok():
            return time.time() - start_time < time_limit

        # Strategy: Multi-start with CMA-ES-like behavior using adaptive step sizes
        # Phase 1: Initial exploration with random starts
        # Phase 2: Local search around best points using adaptive (1+1) or (1+lambda)
        # Phase 3: Polish with smaller steps

        rng = np.random.default_rng(12345)
        
        best_x = None
        best_f = np.inf
        calls = 0

        # Helper: local search using adaptive step size
        def local_search(x_start, f_start, max_calls, initial_sigma=1.0):
            nonlocal calls, best_x, best_f
            
            x = x_start.copy()
            fx = f_start
            sigma = initial_sigma
            calls_used = 0
            
            # Success window for sigma adaptation
            window = []
            window_size = 10
            
            while calls_used < max_calls and calls < budget and _time_ok():
                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                calls_used += 1
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                
                if fy < fx:
                    x = y
                    fx = fy
                    window.append(1)
                else:
                    window.append(0)
                
                # Adapt sigma
                if len(window) > window_size:
                    window.pop(0)
                
                if len(window) == window_size:
                    success_rate = np.mean(window)
                    if success_rate > 0.4:
                        sigma *= 1.2
                    elif success_rate < 0.15:
                        sigma *= 0.8
                    else:
                        sigma *= 1.05  # slight increase
                    sigma = max(sigma, 0.01)
                    sigma = min(sigma, 5.0)
                
                # Early exit if sigma becomes very small
                if sigma < 0.005:
                    break
            
            return x, fx, calls_used

        # Phase 1: Initial sampling - 10 random points
        n_init = 10
        starts = []
        
        for i in range(n_init):
            if not _time_ok() or calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls += 1
            starts.append((x.copy(), f))
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Sort starts by fitness
        starts.sort(key=lambda t: t[1])
        
        # Phase 2: Local search from top starts
        # Allocate budget for local searches
        remaining_calls = budget - calls
        
        # Use top 3 starts for local search
        n_starts_for_search = 3
        calls_per_start = max(50, remaining_calls // (n_starts_for_search + 1))
        
        for i in range(min(n_starts_for_search, len(starts))):
            if not _time_ok() or calls >= budget:
                break
            
            x_start, f_start = starts[i]
            # Use adaptive initial sigma based on current best
            initial_sigma = 1.0 if f_start > best_f else 0.5
            
            x_loc, f_loc, calls_used = local_search(
                x_start, f_start, 
                max_calls=calls_per_start,
                initial_sigma=initial_sigma
            )
        
        # Phase 3: Polish - use remaining budget for fine-tuning
        if best_x is not None and calls < budget and _time_ok():
            remaining = budget - calls
            polish_steps = min(remaining, 100)
            
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            
            for _ in range(polish_steps):
                if not _time_ok() or calls >= budget:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                
                sigma = max(sigma, 0.005)
                sigma = min(sigma, 1.0)
        
        return best_x, best_f
