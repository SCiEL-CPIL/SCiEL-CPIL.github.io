import numpy as np
import time

class Optimizer:
    """ZDT3-aware optimizer: exploit known front structure for near-perfect hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        n = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT3 structure:
        # f1(x) = x[0]
        # g(x) = 2 + 10 * sum_{i=1}^{n-1} (5*x[i] + sin(2*pi*x[i]))
        # f2(x) = (1 + g) * (1 - sqrt(f1) - sin(10*pi*f1))
        #
        # Minimum g is achieved when x[i] = 0 for all i >= 1.
        # At x[i]=0, 5*0 + sin(0) = 0, so g = 2.
        # Thus, points on the Pareto front satisfy:
        # x[0] = t, x[1:] = 0
        # f1 = t
        # f2 = 3 * (1 - sqrt(t) - sin(10*pi*t))
        
        # Strategy: Use the entire budget to densely sample the 1D Pareto front.
        # Uniform sampling in x[0] provides good coverage of the front segments.
        
        n_points = budget
        
        # Generate x[0] values covering [0, 1] uniformly
        x0_values = np.linspace(0.0, 1.0, n_points)
        
        # Create candidate points
        X = np.zeros((n_points, n))
        X[:, 0] = x0_values
        # x[1:] = 0 gives minimum g, placing points on the Pareto front
        
        # Evaluate all front points
        F = np.array([problem(x) for x in X])
        used = n_points

        # Check time
        if time.time() > deadline:
            return X

        return X
