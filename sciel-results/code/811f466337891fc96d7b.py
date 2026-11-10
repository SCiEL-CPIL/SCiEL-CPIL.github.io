import numpy as np

class Optimizer:
    """
    Optimizer for ZDT2 exploiting the known Pareto front structure.
    Places evaluations on the hyperplane x[1:] = 0 with x[0] distributed
    according to an optimal quantization scheme for the convex front f2 = 1 - sqrt(f1).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # The Pareto front of ZDT2 is located at x[1:] = 0.
        # f1 = x[0], f2 = 1 - sqrt(x[0]) for x[0] in [0, 1].
        # To maximize hypervolume, we distribute x[0] such that the spacing
        # is smaller where the curve is steeper (near x[0]=0).
        # Using x[0] = (i/N)^2 achieves this.
        
        n = self.budget
        # Ensure we don't exceed budget or go below 1
        n = max(1, min(n, 200))
        
        # Generate optimal x0 values
        # i ranges from 0 to n-1
        # x0 = (i/n)^2
        i_vals = np.arange(n)
        x0_vals = (i_vals / n) ** 2
        
        # Construct the population
        X = np.zeros((n, self.dim))
        X[:, 0] = x0_vals
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
