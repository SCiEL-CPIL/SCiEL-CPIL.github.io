import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 290

    def __call__(self, func):
        rng = np.random.default_rng(0)
        best_x = None
        best_f = float('inf')
        evals_used = 0
        
        pop_size = 4
        max_no_improve = 30
        
        # We will loop until budget is exhausted or time is out
        while evals_used < self.budget:
            if time.time() > self.deadline:
                break
            
            # Start a new local search (restart)
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            evals_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 2.0
            no_improve = 0
            
            while evals_used < self.budget:
                if time.time() > self.deadline:
                    break
                
                # Generate offspring
                offspring = x + sigma * rng.standard_normal((pop_size, self.dim))
                offspring = np.clip(offspring, -5, 5)
                
                # Evaluate offspring
                min_f_offspring = float('inf')
                best_offspring = None
                for y in offspring:
                    fy = func(y)
                    evals_used += 1
                    if fy < min_f_offspring:
                        min_f_offspring = fy
                        best_offspring = y
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                
                if min_f_offspring < fx:
                    x = best_offspring.copy()
                    fx = min_f_offspring
                    sigma *= 1.2
                    no_improve = 0
                else:
                    sigma *= 0.8
                    no_improve += 1
                    if no_improve >= max_no_improve:
                        # Restart to escape local minimum
                        break

        return best_x, best_f
