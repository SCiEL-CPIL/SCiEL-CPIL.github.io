import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        d = self.dim
        budget = self.budget
        
        # Initial point
        x = np.zeros(d)
        f_x = func(x)
        calls = 1
        
        best_x = x.copy()
        best_f = f_x
        
        # Hooke-Jeeves Parameters
        steps = np.ones(d)
        min_step = 1e-8
        max_step = 10.0
        
        # Main Loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            all_converged = True
            for i in range(d):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                
                if steps[i] < min_step:
                    continue
                
                all_converged = False
                
                # Try positive direction
                x_plus = x.copy()
                x_plus[i] += steps[i]
                x_plus = np.clip(x_plus, -5, 5)
                f_plus = func(x_plus)
                calls += 1
                
                if f_plus < f_x:
                    x = x_plus
                    f_x = f_plus
                    steps[i] = min(steps[i] * 1.5, max_step)
                    if f_x < best_f:
                        best_f = f_x
                        best_x = x.copy()
                else:
                    # Try negative direction
                    x_minus = x.copy()
                    x_minus[i] -= steps[i]
                    x_minus = np.clip(x_minus, -5, 5)
                    f_minus = func(x_minus)
                    calls += 1
                    
                    if f_minus < f_x:
                        x = x_minus
                        f_x = f_minus
                        steps[i] = min(steps[i] * 1.5, max_step)
                        if f_x < best_f:
                            best_f = f_x
                            best_x = x.copy()
                    else:
                        steps[i] *= 0.5
            
            if all_converged:
                break

        return best_x, best_f
