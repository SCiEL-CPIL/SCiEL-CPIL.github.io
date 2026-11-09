"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive sigma and stall detection for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls
            if calls >= self.budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            return fx

        # Strategy: Multi-restart (1+1)-ES with aggressive restarts and final polish.
        # Büche-Rastrigin is multimodal; fast restarts help escape local minima.
        
        # 1. Initial broad exploration
        num_initial_starts = 10
        for i in range(num_initial_starts):
            if calls >= self.budget or time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(self.dim)
            else:
                x = rng.uniform(-4, 4, self.dim)
            
            fx = evaluate(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # 2. Focused local search with frequent restarts
        sigma_min = 0.05
        sigma_max = 5.0
        stall_threshold = 15  # Lower threshold to restart faster
        max_restart_attempts = 30
        
        for attempt in range(max_restart_attempts):
            if calls >= self.budget or time.time() > deadline:
                break
            
            # Decide where to restart from
            if attempt == 0:
                x = best_x.copy() if best_x is not None else rng.uniform(-4, 4, self.dim)
                sigma = 1.0
            elif attempt % 4 == 0:
                # Periodically try a fresh random start
                x = rng.uniform(-4, 4, self.dim)
                sigma = 2.0
            else:
                # Perturb the current best
                x = best_x.copy() + rng.normal(0, 1.0, self.dim)
                x = np.clip(x, -5, 5)
                sigma = 0.5
            
            fx = evaluate(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Local (1+1)-ES with stall detection
            stall_count = 0
            max_local_steps = 60  # Limit steps per local search
            
            for step in range(max_local_steps):
                if calls >= self.budget or time.time() > deadline:
                    break
                
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.2, sigma_max)
                    stall_count = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.7, sigma_min)
                    stall_count += 1
                    if stall_count >= stall_threshold:
                        break
            
            if time.time() > deadline:
                break
        
        # 3. Final polish with Nelder-Mead if budget allows
        if best_x is not None and calls < self.budget and time.time() < deadline:
            try:
                from scipy.optimize import minimize
                # Use remaining budget for maxiter estimate
                # Nelder-Mead in 5D uses ~5*dim evals per iteration? 
                # Let's just run it and let it stop if it hits the limit via a wrapper?
                # Simpler: run a small number of iterations.
                # We can't easily limit evals in scipy minimize without a wrapper that counts.
                # Let's create a wrapper that stops if budget is exceeded.
                eval_count = [0]
                def wrapped_func(x):
                    nonlocal calls
                    if calls >= self.budget:
                        return np.inf
                    x = np.clip(x, -5.0, 5.0)
                    fx = float(func(x))
                    calls += 1
                    return fx

                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
