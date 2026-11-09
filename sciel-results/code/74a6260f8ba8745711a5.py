import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def try_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Multi-start Adaptive Simulated Annealing (ASA)
        # ASA uses a deterministic direction based on the difference between the current point and a reference point,
        # scaled by a temperature that decays over time. This allows it to escape local minima early on
        # and then converge as the temperature drops.
        
        # Number of starts
        n_starts = 5
        budget_per_start = budget // n_starts
        last_start_budget = budget - budget_per_start * (n_starts - 1)
        
        # Initialize best with a few random points to have a reference
        for _ in range(min(10, budget)):
            x0 = rng.uniform(-5, 5, dim)
            fx = try_call(x0)
            if fx is None:
                return best_x, best_f

        # Main ASA loop
        for start_idx in range(n_starts):
            if calls >= budget or time.time() >= deadline:
                break
            
            # Determine starting point for this start
            if start_idx == 0:
                # Start near the best found so far or origin
                x_current = best_x.copy() if best_x is not None else np.zeros(dim)
            else:
                # Restart from a random point or perturbation of best
                if rng.random() < 0.5:
                    x_current = best_x + rng.normal(0, 2.0, dim)
                else:
                    x_current = rng.uniform(-5, 5, dim)
            
            x_current = np.clip(x_current, -5.0, 5.0)
            f_current = try_call(x_current)
            if f_current is None:
                break
            
            # Reference point: initialized as the current point, updated when a better point is found
            x_ref = x_current.copy()
            f_ref = f_current
            
            # Temperature parameters
            # T0 is the initial temperature, T_end is the final temperature
            # We want the temperature to decay exponentially
            T0 = 10.0
            T_end = 1e-3
            # Number of steps for this start
            n_steps = budget_per_start if start_idx < n_starts - 1 else last_start_budget
            n_steps = max(10, n_steps)
            
            # Decay rate
            if n_steps > 1:
                decay_rate = (T_end / T0) ** (1.0 / n_steps)
            else:
                decay_rate = 1.0
                
            T = T0
            
            for step in range(n_steps):
                if calls >= budget or time.time() >= deadline:
                    break
                
                # ASA Direction: d = (x_ref - x_current)
                # If x_current is better than x_ref, x_ref is updated to x_current, so d becomes 0?
                # Standard ASA:
                # x_new = x_current + T * (x_ref - x_current) + noise
                # If f_new < f_current:
                #   x_current = x_new
                #   x_ref = x_current (if f_new < f_ref)
                
                d = x_ref - x_current
                
                # Add noise to escape local minima
                noise = T * rng.standard_normal(dim)
                
                x_candidate = x_current + d + noise
                x_candidate = np.clip(x_candidate, -5.0, 5.0)
                
                f_candidate = try_call(x_candidate)
                if f_candidate is None:
                    break
                
                # Metropolis acceptance criterion
                if f_candidate < f_current:
                    x_current = x_candidate
                    f_current = f_candidate
                    if f_candidate < f_ref:
                        x_ref = x_candidate
                        f_ref = f_candidate
                else:
                    # Accept with probability exp(-(f_candidate - f_current) / T)
                    if rng.random() < np.exp(-(f_candidate - f_current) / T):
                        x_current = x_candidate
                        f_current = f_candidate
                        # Note: x_ref is NOT updated on Metropolis acceptance of a worse point
                        # x_ref is only updated on a new global best (within the start)
                
                # Decay temperature
                T *= decay_rate

        # Final local refinement from the best point found
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma_local = 0.1
            no_improve = 0
            max_no_improve = 30
            
            while calls < budget and time.time() < deadline:
                y = x + sigma_local * rng.standard_normal(dim)
                fy = try_call(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    sigma_local *= 1.05
                    no_improve = 0
                else:
                    sigma_local *= 0.9
                    no_improve += 1
                    
                if no_improve >= max_no_improve:
                    break

        return best_x, best_f
