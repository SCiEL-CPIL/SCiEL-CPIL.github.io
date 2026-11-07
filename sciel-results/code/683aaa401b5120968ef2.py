import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            start_time = time.time()
            deadline = start_time + 290.0
            rng = np.random.default_rng(0)
            
            # Initial point
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            sigma = 2.0
            best_x = x.copy()
            best_fx = fx
            no_improve = 0
            calls = 1
            
            for _ in range(self.budget - 1):
                if time.time() > deadline:
                    break
                if calls >= self.budget:
                    break
                
                # Generate candidate
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fy < best_fx:
                        best_fx = fy
                        best_x = y.copy()
                    no_improve = 0
                    # Increase sigma slightly, but not too fast
                    sigma *= 1.1
                else:
                    no_improve += 1
                    sigma *= 0.9
                    # Keep x as is for the next step? Or revert?
                    # In (1+1), if reject, x stays.
                    
                # Stagnation Restart Logic
                # If we haven't improved in a while, the sigma might be too small
                # or we are in a flat region.
                if no_improve > 40:
                    # Reset sigma to a moderate value
                    sigma = 1.5
                    # Perturb current best to escape local trap
                    # Use a fraction of the remaining budget to decide how aggressive?
                    # Just a small kick
                    kick = 0.5 * rng.standard_normal(self.dim)
                    x = np.clip(best_x + kick, -5, 5)
                    fx = func(x) # This costs a call!
                    calls += 1
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    no_improve = 0
                    if calls >= self.budget:
                        break
            return best_x, best_fx
