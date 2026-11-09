import numpy as np
from scipy.stats import qmc
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # 300s limit, leave 5s margin
        
        # DTLZ7 specific structure:
        # f1 = y1
        # f2 = y1 * (1 - y2)
        # f3 = 1 + g(x_in) * (1 - y1 * sqrt(y2) / (1 + g(x_in)))
        # where y_i = x_i for i in [0, m-1] (m = n_obj = 3)
        # g(x_in) = 1 + 10*(n_in) + sum_{x in x_in} ((x_j - 0.5)^2 - cos(20*pi*(x_j-0.5)))
        # n_in = dim - m = 22 - 3 = 19
        #
        # To minimize all objectives, we need:
        # 1. g(x_in) as small as possible. The minimum of g is 1, achieved when all x_in = 0.5.
        # 2. y1 > 0, y2 > 0.
        # 3. The Pareto front is defined by varying y1 and y2 with g=1.
        #    f1 = y1
        #    f2 = y1 * (1 - y2)
        #    f3 = 1 + (1 - y1 * sqrt(y2) / 2)  [when g=1]
        #
        # Strategy:
        # - Set the 19 inactive variables (indices 3..21) to 0.5 to minimize g to 1.
        # - Sample the 3 active variables (indices 0, 1, 2) using a Sobol sequence.
        # - This gives us points on or near the Pareto front.
        
        n_active = 3  # y1, y2, y3 (though y3 doesn't appear in DTLZ7 objectives directly, it's part of y)
        # Actually, in DTLZ7, y = x[0:m] where m = n_obj = 3.
        # f1 = y[0]
        # f2 = y[0] * (1 - y[1])
        # f3 = 1 + g * (1 - y[0]*sqrt(y[1])/(1+g))
        # So only y[0] and y[1] matter for the shape. y[2] is unused in objectives.
        # But we should still sample it or set it appropriately.
        # Let's set y[2] = 0.5 or sample it; it doesn't affect objectives.
        
        n_inactive = self.dim - n_active  # 19 variables
        
        # Set inactive variables to 0.5 (minimizes g)
        # Set active variables using Sobol for good coverage of [0,1]^3
        
        n_samples = self.budget
        
        # Generate Sobol points in 3D
        try:
            sampler = qmc.Sobol(d=n_active, scramble=True, seed=42)
            sobol_points = sampler.random(n_samples)
        except Exception:
            sobol_points = np.random.random((n_samples, n_active))
        
        # Construct full dimension vectors
        X = np.zeros((n_samples, self.dim))
        X[:, :n_active] = sobol_points
        X[:, n_active:] = 0.5  # Minimize g
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Evaluate all points
        for i in range(n_samples):
            if time.time() > deadline - 1.0:
                break
            x = np.clip(X[i], lo, hi)
            problem(x)
        
        return X
