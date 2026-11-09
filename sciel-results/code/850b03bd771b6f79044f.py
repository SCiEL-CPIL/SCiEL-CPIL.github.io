import numpy as np
import scipy.optimize
import time

class Optimizer:
    """
    Hybrid optimizer combining (1+lambda)-ES for global search with Nelder-Mead for local refinement.
    Phase 1: Stratified multi-start exploration to locate promising regions.
    Phase 2: Population-based (1+lambda)-ES with adaptive step size and restart mechanism.
    Phase 3: Nelder-Mead for high-precision local refinement.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit

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

        # Phase 1: Stratified multi-start exploration
        # Use quasi-random-like sampling for better coverage than pure random
        n_init = min(20, budget // 10)
        
        for i in range(n_init):
            if total_calls >= budget or time.time() > deadline:
                break
            
            if i < n_init // 4:
                x = rng.uniform(-4.5, 4.5, dim)
            elif i < n_init // 2:
                x = rng.uniform(-3.0, 3.0, dim)
            elif i < 3 * n_init // 4:
                x = rng.uniform(-1.5, 1.5, dim)
            else:
                x = rng.uniform(-0.5, 0.5, dim)
            
            evaluate(x)

        if total_calls >= budget or time.time() > deadline:
            return global_best_x, global_best_f

        # Phase 2: Population-based (1+lambda)-ES
        # Allocate 60% of remaining budget to ES, 40% to NM
        remaining_budget = budget - total_calls
        es_budget = int(remaining_budget * 0.6)
        nm_budget = remaining_budget - es_budget
        
        x = global_best_x.copy() if global_best_x is not None else rng.uniform(-1, 1, dim)
        fx = global_best_f
        
        sigma = 1.5  # Initial step size
        pop_size = 6
        successes = []
        window_size = 12
        stagnation = 0
        max_stagnation = 30
        
        while total_calls < budget and time.time() < deadline:
            # Check if we should switch to NM phase
            if total_calls >= (budget - nm_budget):
                break
                
            # Generate population of offspring
            improvements = 0
            for j in range(pop_size):
                if total_calls >= budget or time.time() > deadline:
                    break
                if total_calls >= (budget - nm_budget):
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, lower, upper)
                fy = evaluate(y)

                if fy < fx:
                    improvements += 1
                    x, fx = y, fy
                    successes.append(1)
                    stagnation = 0
                else:
                    successes.append(0)
                    stagnation += 1

            # Trim success window
            if len(successes) > window_size:
                successes = successes[-window_size:]

            # Adaptive sigma adjustment
            if len(successes) >= window_size:
                rate = sum(successes) / window_size
                if rate > 0.35:
                    sigma *= 1.25
                elif rate > 0.20:
                    sigma *= 1.10
                elif rate < 0.10:
                    sigma *= 0.80
                elif rate < 0.20:
                    sigma *= 0.92
                else:
                    sigma *= 0.97

            sigma = np.clip(sigma, 0.005, 5.0)

            # Restart mechanism
            if stagnation > max_stagnation:
                if sigma < 0.15:
                    # Likely in a sharp local min, restart from best with larger sigma
                    x = global_best_x.copy()
                    fx = global_best_f
                    sigma = 2.5
                    stagnation = 0
                    successes = []
                else:
                    # Try larger jumps to escape plateau
                    found_improvement = False
                    for _ in range(min(5, budget - total_calls)):
                        if total_calls >= budget or time.time() > deadline:
                            break
                        if total_calls >= (budget - nm_budget):
                            break
                        jump_scale = sigma * rng.uniform(1.5, 3.0)
                        y = x + jump_scale * rng.standard_normal(dim)
                        y = np.clip(y, lower, upper)
                        fy = evaluate(y)
                        if fy < fx:
                            x, fx = y, fy
                            stagnation = 0
                            found_improvement = True
                            break
                    
                    if not found_improvement:
                        x = global_best_x.copy()
                        fx = global_best_f
                        sigma = 3.0
                    stagnation = 0
                    successes = []

        # Phase 3: Nelder-Mead for local refinement
        if total_calls < budget and time.time() < deadline and global_best_x is not None:
            x0 = global_best_x.copy()
            
            # Create initial simplex around best point with small step
            simplex = [x0.copy()]
            step = 0.1
            for i in range(dim):
                p = x0.copy()
                p[i] += step
                p = np.clip(p, lower, upper)
                simplex.append(p)
            
            nm_calls = 0
            nm_best_f = fx
            
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
            try:
                global_best_f = func(np.clip(global_best_x, lower, upper))
            except Exception:
                global_best_f = 1e6
        
        return global_best_x, global_best_f
