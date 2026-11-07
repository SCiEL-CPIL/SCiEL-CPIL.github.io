import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or not time_ok():
                return None
            x = np.clip(x, -5.0, 5.0)
            f_val = func(x)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global exploration with structured + random sampling
        # Büche-Rastrigin has many local minima; we need broad coverage
        # Use a mix of uniform random and quasi-random (Halton-like) points
        
        n_global = 40  # Use 40 calls for global phase
        
        # Add origin
        f0 = call_func(np.zeros(dim))
        if f0 is None:
            return best_x, best_f

        # Add axis-aligned points
        for i in range(dim):
            for sign in [-1, 1]:
                x = np.zeros(dim)
                x[i] = sign * 3.0
                f = call_func(x)
                if f is None:
                    break
            if f is None:
                break
        if calls_made >= budget or not time_ok():
            return best_x, best_f

        # Random sampling in [-4, 4]^5
        for _ in range(n_global - 1 - 2 * dim):
            if calls_made >= budget or not time_ok():
                break
            x = rng.uniform(-4.0, 4.0, dim)
            f = call_func(x)
            if f is None:
                break

        if calls_made >= budget or not time_ok():
            return best_x, best_f

        # Phase 2: Multi-start (1+1)-ES
        # Büche-Rastrigin has many local minima, so multiple restarts help
        # Divide remaining budget among restarts
        
        remaining = budget - calls_made
        n_restarts = 5
        budget_per_restart = max(10, remaining // n_restarts)
        
        for restart in range(n_restarts):
            if calls_made >= budget or not time_ok():
                break
            
            # Initialize this restart
            if restart == 0:
                # Start from best found so far
                x = best_x.copy()
            else:
                # Mix: sometimes perturb best, sometimes sample fresh
                if rng.random() < 0.4:
                    x = best_x + rng.normal(0, 1.5, dim)
                else:
                    x = rng.uniform(-4.0, 4.0, dim)
                x = np.clip(x, -5.0, 5.0)
            
            f = call_func(x)
            if f is None:
                break
            
            local_best_f = f
            local_best_x = x.copy()
            sigma = 1.0
            stagnant = 0
            stagnation_limit = 30
            
            # (1+1)-ES loop for this restart
            while calls_made < budget and time_ok():
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5.0, 5.0)
                
                f_trial = call_func(x_trial)
                if f_trial is None:
                    break
                
                if f_trial < local_best_f:
                    # Success: increase sigma slightly
                    sigma *= 1.05
                    stagnant = 0
                    local_best_f = f_trial
                    local_best_x = x_trial.copy()
                    x = x_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = x_trial.copy()
                else:
                    # Failure: decrease sigma
                    sigma *= 0.9
                    stagnant += 1
                
                # Stagnation check
                if stagnant >= stagnation_limit:
                    break
                
                # Minimum sigma guard
                if sigma < 0.005:
                    break

        return best_x, best_f
