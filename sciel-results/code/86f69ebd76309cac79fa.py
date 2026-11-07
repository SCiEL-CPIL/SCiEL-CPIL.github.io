import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.start_time = None

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        deadline = self.start_time + 295.0 # 5s margin
        
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        self.call_count += 1
        fx = func(x)
        sigma = 2.0
        
        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            self.call_count += 1
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87
        return x, fx
