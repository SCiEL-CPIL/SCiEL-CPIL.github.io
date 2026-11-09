import numpy as np
import scipy.optimize
import time

class Optimizer:
    """
    Hybrid optimizer combining (1+1)-ES for global search with Nelder-Mead for local refinement.
    Phase 1: (1+1)-ES with adaptive step size to locate the global basin.
    Phase 2: Nelder-Mead to refine the solution to high precision.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget
        lower, upper = self.bounds
        rng = np.random.default_rng(42)

        global_best_x = None
        global_best_f = float('inf')
        total_calls = 0

        def evaluate(x):
            """Evaluate a point, tracking calls and updating global best."""
            nonlocal total_calls, global_best_f, global_best_x
            if total_calls >= budget or time.time() > deadline:
                return 1e6
            x_clipped = np.clip(x, lower, upper)
            try:
                f_val = func(x_clipped)
            except Exception:
                f_val = 1e6
            total_calls += 1
            if np.isnan(f_val) or np.isinf(f_val):
                return 1e6
            if f_val < global_best_f:
                global_best_f = f_val
                global_best_x = x_clipped.copy()
            return f_val

        # Phase 1: (1+1)-ES for global search
        # Use 60% of budget for global search
        es_budget = int(budget * 0.6)
        es_calls = 0
        sigma = 1.0  # Initial step size
        x = rng.uniform(-4.0, 4.0, dim)
        f_x = evaluate(x)
        
        # Adaptive step size parameters
        success_count = 0
        total_attempts = 0
        sigma_min = 0.01
        sigma_max = 3.0
        
        while es_calls < es_budget and total_calls < budget and time.time() < deadline:
            # Generate mutation
            z = rng.standard_normal(dim)
            x_trial = x + sigma * z
            
            f_trial = evaluate(x_trial)
            es_calls += 1
            
            if f_trial < f_x:
                # Success: move to trial point
                x = x_trial
                f_x = f_trial
                success_count += 1
            total_attempts += 1
            
            # Adapt step size every 10 attempts
            if total_attempts % 10 == 0:
                success_rate = success_count / total_attempts
                if success_rate > 0.35:
                    sigma = min(sigma * 1.5, sigma_max)
                elif success_rate < 0.15:
                    sigma = max(sigma * 0.7, sigma_min)
                success_count = 0
                total_attempts = 0
        
        # Phase 2: Nelder-Mead for local refinement
        # Use remaining budget for local refinement
        nm_budget = budget - total_calls
        
        if nm_budget > 5 and global_best_x is not None:
            x0 = global_best_x.copy()
            
            # Create initial simplex around best point
            simplex = [x0.copy()]
            step = max(sigma * 0.5, 0.1)
            for i in range(dim):
                p = x0.copy()
                p[i] += step
                p = np.clip(p, lower, upper)
                simplex.append(p)
            
            nm_calls = 0
            nm_best_f = f_x
            
            def nm_func(x):
                nonlocal nm_calls, nm_best_f
                if total_calls >= budget or time.time() > deadline:
                    return 1e6
                x_clipped = np.clip(x, lower, upper)
                try:
                    f_val = func(x_clipped)
                except Exception:
                    f_val = 1e6
                total_calls += 1
                nm_calls += 1
                if np.isnan(f_val) or np.isinf(f_val):
                    return 1e6
                if f_val < nm_best_f:
                    nm_best_f = f_val
                if f_val < global_best_f:
                    global_best_f = f_val
                    global_best_x = x_clipped.copy()
                return f_val
            
            def nm_callback(xk):
                if nm_calls >= nm_budget or total_calls >= budget or time.time() > deadline:
                    return True
                return False
            
            try:
                scipy.optimize.minimize(
                    nm_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True,
                        'initial_simplex': simplex
                    },
                    callback=nm_callback
                )
            except Exception:
                pass
        
        # Fallback if no valid evaluation was made
        if global_best_x is None:
            global_best_x = np.zeros(dim)
            global_best_f = func(np.clip(global_best_x, lower, upper))
        
        return global_best_x, global_best_f
