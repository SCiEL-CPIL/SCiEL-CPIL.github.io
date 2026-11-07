import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300
        
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        calls = 1
        lam = 5
        
        while calls < self.budget:
            if time.time() > deadline - 10:
                break
                
            # Generate lambda candidates
            candidates = []
            for _ in range(lam):
                if calls >= self.budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1
                candidates.append((fy, y))
                
            if not candidates:
                break
                
            # Pick the best
            candidates.sort(key=lambda t: t[0])
            fy_best, y_best = candidates[0]
            
            if fy_best < fx:
                x, fx = y_best, fy_best
                sigma *= 1.2
            else:
                sigma *= 0.9
                
        return x, fx
