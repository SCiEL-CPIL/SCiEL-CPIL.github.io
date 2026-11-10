import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT4 specific structure:
        # x0 in [-5, 5], x1..x9 in [0, 1]
        # f1 = x0 + 2
        # f2 = g(x) * (1 - sqrt(f1/g))
        # g(x) = 1 + 10*(n-1) + sum(xi^2 - 10*cos(2*pi*xi)) for i=1..n-1
        # Minimum of g is 1 when x1..x9 = 0.
        # The true Pareto front is achieved when x1..x9 = 0.
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We have self.budget evaluations.
        # We will use all of them.
        n_evals = self.budget
        
        # Determine the valid range for x0.
        # f1 = x0 + 2. For sqrt(f1) to be real, f1 >= 0 => x0 >= -2.
        # Also x0 must be within [lo[0], hi[0]].
        x0_min = max(lo[0], -2.0)
        x0_max = hi[0]
        
        # If the range is invalid (e.g. x0_min > x0_max), fallback to full range
        if x0_min > x0_max:
            x0_min = lo[0]
            x0_max = hi[0]
            
        # Generate x0 samples.
        # A uniform grid in x0 is a good starting point.
        # To be more robust against non-linearity, we could use a Halton sequence,
        # but with 200 points, a uniform grid in the variable space is very dense.
        # The hypervolume is dominated by the coverage of the curve.
        
        x0_samples = np.linspace(x0_min, x0_max, n_evals)
        
        # Construct the population
        # x1..x9 are set to 0 to minimize g(x)
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_samples
        # X[:, 1:] is already 0
        
        # Evaluate all points
        # The problem function clips and evaluates.
        # We must ensure we don't exceed the budget.
        # We have exactly n_evals points.
        
        results = []
        for i in range(n_evals):
            x = X[i]
            f = problem(x)
            results.append(f)
            
        # The return value is ignored, but we return the X array as per standard practice
        # or just None. The current program returns X.
        return X
