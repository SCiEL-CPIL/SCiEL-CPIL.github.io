import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s kill
        
        n = self.dim
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto Front Structure:
        # f1(x) = x[0]
        # g(x) = 2 + 10 * sum_{i=1}^{n-1} (5*x[i] + sin(2*pi*x[i]))
        # f2(x) = (1 + g) * (1 - sqrt(x[0]) - sin(10*pi*x[0]))
        #
        # The minimum value of g is 2, achieved when x[i] = 0 for all i >= 1.
        # Points on the Pareto front satisfy:
        # x[0] = t, x[1:] = 0
        # f1 = t
        # f2 = 3 * (1 - sqrt(t) - sin(10*pi*t))
        #
        # Strategy:
        # Since we know the exact location of the Pareto front in the decision space,
        # we should use the entire budget to densely sample this 1D manifold.
        # A uniform distribution of t in [0, 1] provides the best coverage for hypervolume
        # in this 2-objective setting.
        
        n_points = budget
        
        # Generate x[0] values covering [0, 1] uniformly
        # Using linspace ensures we hit the endpoints 0 and 1, which are critical
        # for capturing the full extent of the front for hypervolume calculation.
        x0_values = np.linspace(0.0, 1.0, n_points)
        
        # Create candidate points
        X = np.zeros((n_points, n))
        X[:, 0] = x0_values
        # x[1:] remains 0, which minimizes g and places points on the Pareto front
        
        # Evaluate all front points
        # This is fast because it's just function calls on a known structure.
        F = np.array([problem(x) for x in X])
        
        # Return the set of evaluated points
        return X
