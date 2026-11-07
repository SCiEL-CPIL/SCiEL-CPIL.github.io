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
        deadline = self.start_time + 295.0  # 5s margin

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        self.call_count += 1
        fx = func(x)
        sigma = 2.0
        best_x = x.copy()
        best_fx = fx
        no_improve = 0
        max_no_improve = 60

        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            self.call_count += 1
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                no_improve = 0
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                sigma *= 0.9
                no_improve += 1

            if no_improve >= max_no_improve and self.call_count < self.budget - 10:
                # Restart from a new random point
                x = rng.uniform(-5, 5, self.dim)
                self.call_count += 1
                fx = func(x)
                sigma = 2.0
                no_improve = 0
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        return best_x, best_fx
