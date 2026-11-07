"""Optimizer for BBOB f6 (Attractive sector) using Differential Evolution and Nelder-Mead polish."""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """DE + NM polish optimizer for f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0
        
        best_x = None
        best_f = np.inf
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_x = x_clipped.copy()
                best_f = f
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Differential Evolution
        pop_size = 12
        F = 0.8
        CR = 0.9
        
        # Initialize population
        pop = np.empty((pop_size, dim))
        pop[0] = 0.0
        pop[1:] = rng.uniform(-4.0, 4.0, (pop_size - 1, dim))
        
        # Evaluate initial population
        pop_f = np.empty(pop_size)
        for i in range(pop_size):
            if calls >= budget or time.time() > deadline:
                break
            pop_f[i] = eval_func(pop[i])
        
        # Handle case where we couldn't evaluate all
        if calls < pop_size:
             for i in range(calls, pop_size):
                 pop_f[i] = np.inf
        else:
             # Ensure all are valid
             pass

        # DE Loop
        # Each generation costs pop_size evaluations
        while calls + pop_size <= budget and time.time() < deadline:
            # Generate trial vectors
            trials = np.empty_like(pop)
            for i in range(pop_size):
                # Select distinct random indices
                r1, r2 = rng.choice(pop_size, 2, replace=False)
                while r1 == i or r2 == i:
                    r1, r2 = rng.choice(pop_size, 2, replace=False)
                
                # DE/rand/1
                v = pop[r1] + F * (pop[r2] - pop[r1])
                
                # Crossover
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trials[i, j] = v[j]
                    else:
                        trials[i, j] = pop[i, j]
                
                # Clip
                trials[i] = np.clip(trials[i], -5.0, 5.0)
            
            # Evaluate trials
            trial_f = np.empty(pop_size)
            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break
                trial_f[i] = eval_func(trials[i])
            
            # If budget ran out during evaluation, update best from what we have
            if calls >= budget or time.time() > deadline:
                 for i in range(pop_size):
                     if not np.isnan(trial_f[i]):
                         if trial_f[i] < best_f:
                             best_f = trial_f[i]
                             best_x = trials[i].copy()
                 break

            # Selection
            for i in range(pop_size):
                if trial_f[i] < pop_f[i]:
                    pop[i] = trials[i]
                    pop_f[i] = trial_f[i]
            
            # Update global best
            current_best_idx = np.argmin(pop_f)
            if pop_f[current_best_idx] < best_f:
                best_f = pop_f[current_best_idx]
                best_x = pop[current_best_idx].copy()

        # Phase 2: Local Polish with Nelder-Mead
        remaining = budget - calls
        if remaining > 20 and best_x is not None and time.time() < deadline:
            # NM uses approx dim+1 calls per iteration
            nm_budget = int(remaining * 0.8)
            max_iter = max(10, nm_budget // (dim + 1))
            
            def objective(z):
                return eval_func(z)
            
            try:
                res = minimize(
                    objective, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': max_iter, 
                        'xatol': 1e-8, 
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
                    best_f = res.fun
            except Exception:
                pass
        
        return best_x, best_f
