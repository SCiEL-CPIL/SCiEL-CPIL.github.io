import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        deadline = time.time() + 280
        calls = 0

        def safe_call(x):
            nonlocal calls
            if time.time() > deadline:
                return None
            calls += 1
            return func(x)

        x0 = np.zeros(dim)
        f0 = safe_call(x0)
        if f0 is None:
            return x0, f0

        delta = 1.0
        a = np.zeros(dim)
        
        for i in range(dim):
            if time.time() > deadline:
                break
            xi = np.zeros(dim)
            xi[i] = delta
            fi = safe_call(xi)
            if fi is None:
                break
            a[i] = (fi - f0) / delta

        # Construct optimal vertex
        x_opt = np.zeros(dim)
        for i in range(dim):
            if a[i] > 0:
                x_opt[i] = -5.0
            elif a[i] < 0:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0

        f_opt_est = safe_call(x_opt)
        if f_opt_est is None:
            return x0, f0

        return x_opt, f_opt_est
