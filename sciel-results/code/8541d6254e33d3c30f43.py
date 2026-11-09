import numpy as np
import time

class Optimizer:
    """Multi-scale coordinate descent for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        best_x = None
        best_f = float('inf')
        calls = 0

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Multi-scale coordinate descent
        # Start with large steps and shrink geometrically
        initial_steps = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625]
        
        # Start from center (likely near minimum for Lunacek)
        x = np.zeros(self.dim)
        fx = func(x)
        calls += 1
        update_best(x, fx)

        for step in initial_steps:
            if calls >= self.budget or time.time() > deadline:
                break
            
            # Coordinate descent passes
            for _ in range(3):  # Multiple passes per step size
                if calls >= self.budget or time.time() > deadline:
                    break
                
                for d in range(self.dim):
                    if calls >= self.budget or time.time() > deadline:
                        break
                    
                    # Try positive and negative directions
                    for sign in [1, -1]:
                        if calls >= self.budget or time.time() > deadline:
                            break
                        
                        trial = x.copy()
                        trial[d] += sign * step
                        trial = np.clip(trial, -5, 5)
                        
                        # Skip if same as current
                        if np.allclose(trial, x, atol=1e-12):
                            continue
                        
                        f_trial = func(trial)
                        calls += 1
                        
                        if f_trial < fx:
                            x, fx = trial, f_trial
                            update_best(x, fx)
                            # If improved, try to continue in same direction
                            # but break to restart coordinate loop with new x
                            break

        # Phase 2: Fine-tuning with smaller steps
        fine_steps = [0.0005, 0.00025, 0.000125, 0.0000625]
        for step in fine_steps:
            if calls >= self.budget or time.time() > deadline:
                break
            
            for d in range(self.dim):
                if calls >= self.budget or time.time() > deadline:
                    break
                
                for sign in [1, -1]:
                    if calls >= self.budget or time.time() > deadline:
                        break
                    
                    trial = x.copy()
                    trial[d] += sign * step
                    trial = np.clip(trial, -5, 5)
                    
                    if np.allclose(trial, x, atol=1e-12):
                        continue
                    
                    f_trial = func(trial)
                    calls += 1
                    
                    if f_trial < fx:
                        x, fx = trial, f_trial
                        update_best(x, fx)
                        break

        # Phase 3: If budget remains, do a broader search
        if calls < self.budget and time.time() < deadline:
            # Try a few random perturbations from best point
            rng = np.random.default_rng(123)
            for _ in range(10):
                if calls >= self.budget or time.time() > deadline:
                    break
                
                for scale in [0.5, 0.2, 0.1]:
                    if calls >= self.budget or time.time() > deadline:
                        break
                    
                    trial = best_x + scale * rng.standard_normal(self.dim)
                    trial = np.clip(trial, -5, 5)
                    f_trial = func(trial)
                    calls += 1
                    
                    if f_trial < best_f:
                        update_best(trial, f_trial)

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
