import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Initialize at the box center for maximum early convergence speed
        x = np.zeros(dim)
        fx = func(x)
        calls = 1

        best_x = x.copy()
        best_f = fx

        # (1+1)-ES with proven adaptation rates
        sigma = 2.0  # Start with a reasonably large step

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate one offspring
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)

            fy = func(y)
            calls += 1

            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.5
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.87
                if sigma < 1e-10:
                    break

        return best_x, best_f
