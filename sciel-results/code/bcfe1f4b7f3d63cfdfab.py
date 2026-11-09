import scipy
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= budget:
                raise StopIteration("Budget exceeded")
            
            x = np.asarray(x)
            f = func(x)
            calls += 1
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Global Search with (1+1)-ES and Restarts
        # Use 60% of budget for global exploration - leave more room for local refinement
        global_budget = int(budget * 0.60)
        
        # Initial point
        x = rng.uniform(-5, 5, dim)
        try:
            fx = safe_func(x)
        except (TimeoutError, StopIteration):
            return best_x, best_f
            
        sigma = 1.0
        stagnation = 0
        max_stagnation = 25
        min_sigma = 0.005
        max_sigma = 10.0
        restart_count = 0
        max_restarts = 8

        for _ in range(global_budget):
            if calls >= budget or time.time() > deadline:
                break
                
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            
            try:
                fy = safe_func(y)
            except (TimeoutError, StopIteration):
                break
                
            if fy < fx:
                x, fx = y, fy
                # Successful step: increase sigma slightly
                sigma = min(sigma * 1.10, max_sigma)
                stagnation = 0
            else:
                # Failed step: decrease sigma
                sigma = max(sigma * 0.90, min_sigma)
                stagnation += 1
                
                # If stagnating, restart with better strategy
                if stagnation > max_stagnation and restart_count < max_restarts:
                    restart_count += 1
                    # Restart from diverse point
                    if best_x is not None:
                        # 40% chance to restart from best with small perturbation, 60% from diverse
                        if rng.random() < 0.4:
                            x = best_x + 1.5 * rng.standard_normal(dim)
                        else:
                            # Diverse restart: perturb best or use pure random
                            if rng.random() < 0.6:
                                x = best_x + 3.0 * rng.standard_normal(dim)
                            else:
                                x = rng.uniform(-5, 5, dim)
                        x = np.clip(x, -5, 5)
                        try:
                            fx = safe_func(x)
                        except (TimeoutError, StopIteration):
                            break
                    else:
                        x = rng.uniform(-5, 5, dim)
                        try:
                            fx = safe_func(x)
                        except (TimeoutError, StopIteration):
                            break
                    
                    # Reset parameters for fresh start
                    sigma = 2.0
                    stagnation = 0
                    max_stagnation = max(10, max_stagnation // 2)

        # Phase 2: Local Refinement using Nelder-Mead
        # Nelder-Mead is derivative-free and robust for non-smooth functions like Schwefel
        if calls < budget and time.time() < deadline and best_x is not None:
            remaining_calls = budget - calls
            if remaining_calls > 10:
                # Nelder-Mead: use a fraction of remaining budget
                # Each iteration uses ~dim+1 calls, so max_iter should be conservative
                max_iter = min(remaining_calls // (dim + 2), 35)
                max_iter = max(5, max_iter)
                
                try:
                    # Initialize simplex around best_x with reasonable spread
                    x0 = best_x.copy()
                    
                    # Create a well-scaled initial simplex
                    simplex = [x0.copy()]
                    for i in range(dim):
                        point = x0.copy()
                        point[i] += 0.1
                        point = np.clip(point, -5, 5)
                        simplex.append(point)
                    
                    result = scipy.optimize.minimize(
                        safe_func, 
                        x0, 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': max_iter,
                            'xatol': 1e-8,
                            'fatol': 1e-8,
                            'adaptive': True,
                            'initial_simplex': np.array(simplex)
                        }
                    )
                except (TimeoutError, StopIteration, Exception):
                    pass
        
        # Phase 3: Final High-Precision Polish with CMA-ES-inspired adaptive (1+1)-ES
        # Uses success-rate-based step size adaptation for better convergence
        if calls < budget and time.time() < deadline and best_x is not None:
            remaining_calls = budget - calls
            if remaining_calls > 5:
                x = best_x.copy()
                fx = best_f  # best_f is already the value at best_x
                
                # CMA-ES-inspired: track success rate for step size adaptation
                sigma = 0.03
                success_window = []
                window_size = 8
                target_success_rate = 0.15
                no_improve_count = 0
                max_no_improve = 20
                
                for _ in range(remaining_calls):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    
                    try:
                        fy = safe_func(y)
                    except (TimeoutError, StopIteration):
                        break
                    
                    success = fy < fx
                    success_window.append(1 if success else 0)
                    if len(success_window) > window_size:
                        success_window.pop(0)
                    
                    if success:
                        x, fx = y, fy
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    # Adapt sigma based on success rate over recent window
                    if len(success_window) >= window_size:
                        recent_rate = sum(success_window) / len(success_window)
                        if recent_rate > target_success_rate:
                            sigma = min(sigma * 1.08, 0.5)
                        elif recent_rate < target_success_rate * 0.4:
                            sigma = max(sigma * 0.85, 1e-7)
                        else:
                            sigma = max(sigma * 0.97, 1e-7)
                    
                    # If no improvement for a while or sigma becomes very small, we've likely converged
                    if sigma < 1e-6 or no_improve_count > max_no_improve:
                        break

        return best_x, best_f
