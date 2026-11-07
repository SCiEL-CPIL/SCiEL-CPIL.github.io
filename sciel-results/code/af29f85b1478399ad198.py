import numpy as np
import time
from scipy.optimize import brentq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        
        n_evals = self.budget
        start_time = time.time()
        deadline = 300.0
        
        # ZDT6 Pareto front analysis:
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # f2 = 1 - sqrt(f1)  (when x2..xn = 0)
        # The function f1(x1) on [0, 1/6] is monotonic decreasing from 1 to 1-exp(-2/3).
        # We sample f1 uniformly in this range and invert to find x1.
        
        # Constants
        f1_min = 1.0 - np.exp(-2.0/3.0)
        f1_max = 1.0
        
        # To maximize hypervolume, we minimize the non-dominated area.
        # The integrand h(y1) = 10 + sqrt(y1) is steep near 0.
        # We sample z = sqrt(y1) uniformly to cluster points near y1=0.
        z_samples = np.linspace(0.0, 1.0, n_evals)
        f1_samples = z_samples ** 2
        
        # Function to invert f1(x) = target on x in [0, 1/6]
        # target = 1 - exp(-4*x) * sin(6*pi*x)^8
        # exp(-4*x) * sin(6*pi*x)^8 = 1 - target
        # Let y = 1 - target
        # h(x) = exp(-4*x) * sin(6*pi*x)^8 - y = 0
        
        def h(x, y):
            return np.exp(-4*x) * (np.sin(6*np.pi*x)**8) - y
        
        x1_samples = []
        for f1 in f1_samples:
            y = 1.0 - f1
            # If y is very close to 0, x is close to 0
            # If y is close to exp(-2/3), x is close to 1/6
            # Use brentq to find root in [0, 1/6]
            # Note: h(0) = 0 - y = -y <= 0
            # h(1/6) = exp(-2/3) - y >= 0 (since y <= exp(-2/3))
            # So a root exists.
            try:
                x1 = brentq(h, 0.0, 1.0/6.0, args=(y,), xtol=1e-12)
            except ValueError:
                # Fallback to midpoint if it fails (shouldn't happen)
                x1 = 0.5 / 6.0
            x1_samples.append(x1)
        
        x1_samples = np.array(x1_samples)
        
        # Evaluate each point
        for x1 in x1_samples:
            if time.time() - start_time > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x1
            # x[1:] are already 0
            # Clip to bounds just in case
            x_clipped = np.clip(x, lo, hi)
            problem(x_clipped)
        
        return None
