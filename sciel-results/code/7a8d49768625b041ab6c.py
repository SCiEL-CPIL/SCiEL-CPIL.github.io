"""Combined program: (1+1)-ES with Restarts + Nelder-Mead Refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """
    Hybrid optimizer:
    1. Global phase: (1+1)-ES with adaptive step size and restarts from best.
    2. Local phase: Nelder-Mead refinement on the best point found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, use 280s to be safe
        deadline = start_time + 280
        
        n = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= self.budget:
                raise StopIteration("Budget exceeded")
            
            x_arr = np.asarray(x)
            f_val = func(x_arr)
            calls += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_arr.copy()
            return f_val

        rng = np.random.default_rng(0)
        
        # Phase 1: Global Search with (1+1)-ES and Restarts
        # Allocate 65% of budget to global exploration to find a good basin
        global_budget = int(self.budget * 0.65)
        
        # Evaluate multiple initial points to get a better starting basin
        n_initial = min(10, global_budget // 5)
        initial_points = rng.uniform(-5, 5, (n_initial, n))
        
        best_init_idx = 0
        for i in range(n_initial):
            if calls >= self.budget or time.time() > deadline:
                break
            try:
                fi = safe_func(initial_points[i])
                if i == 0 or fi < best_f:
                    best_init_idx = i
            except (TimeoutError, StopIteration):
                break
        
        # Start from the best initial point
        x = initial_points[best_init_idx].copy()
        try:
            fx = safe_func(x)
        except (TimeoutError, StopIteration):
            return best_x, best_f
            
        sigma = 1.5
        success_count = 0
        trials = 0
        stagnation = 0
        max_stagnation = 40
        
        while calls < self.budget and time.time() < deadline:
            # Generate candidate
            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            
            try:
                fy = safe_func(y)
            except (TimeoutError, StopIteration):
                break
            
            trials += 1
            if fy < fx:
                x, fx = y, fy
                success_count += 1
                stagnation = 0
            else:
                stagnation += 1
            
            # Adapt step size using 1/5 rule (CMA-ES style)
            if trials >= 10:
                success_rate = success_count / trials
                if success_rate > 0.35:
                    sigma *= 1.05
                elif success_rate < 0.15:
                    sigma *= 0.95
                else:
                    sigma *= 1.01
                success_count = 0
                trials = 0
            
            # If stagnating, restart from global best with diversification
            if stagnation > max_stagnation:
                if best_x is not None:
                    # Diversify: move best point in a random direction
                    direction = rng.standard_normal(n)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    x = best_x + sigma * 2.0 * direction
                    x = np.clip(x, -5, 5)
                else:
                    x = rng.uniform(-5, 5, n)
                    try:
                        fx = safe_func(x)
                    except (TimeoutError, StopIteration):
                        break
                fx = safe_func(x) if best_x is not None else fx
                sigma = max(0.5, sigma * 1.5)
                stagnation = 0
                # Reduce threshold for more aggressive restarts later
                max_stagnation = max(15, max_stagnation - 5)

        # Phase 2: Local Refinement with Nelder-Mead
        # Use remaining budget for local optimization
        if calls < self.budget and time.time() < deadline and best_x is not None:
            remaining_calls = self.budget - calls
            # Try Nelder-Mead from the best point
            max_iter = max(10, remaining_calls // (n + 2))
            
            try:
                # Nelder-Mead is derivative-free and good for final polish
                result = scipy.optimize.minimize(
                    safe_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
            except (TimeoutError, StopIteration, Exception):
                pass # Keep best from Phase 1 if this fails
            
            # If we still have budget, try a second NM from a perturbed best
            if calls < self.budget and time.time() < deadline and best_x is not None:
                remaining_calls = self.budget - calls
                if remaining_calls > (n + 2) * 5:
                    # Perturb best point slightly
                    perturbation = 0.1 * rng.standard_normal(n)
                    x2 = np.clip(best_x + perturbation, -5, 5)
                    max_iter2 = max(5, remaining_calls // (n + 2))
                    try:
                        result2 = scipy.optimize.minimize(
                            safe_func, 
                            x2, 
                            method='Nelder-Mead', 
                            options={
                                'maxiter': max_iter2,
                                'xatol': 1e-6,
                                'fatol': 1e-6,
                                'adaptive': True
                            }
                        )
                    except (TimeoutError, StopIteration, Exception):
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
