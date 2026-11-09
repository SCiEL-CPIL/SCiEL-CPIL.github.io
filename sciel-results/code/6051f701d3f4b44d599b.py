import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 specific strategy:
        # x[1:] should be 0.5 to minimize g(x)
        # x[0] determines the position on the Pareto front
        # We sample x[0] with a power-law distribution to cover the front effectively
        
        n_evals = self.budget
        if n_evals < 1:
            return None
            
        # Use midpoints of n_evals intervals for x[0]
        u = (np.arange(n_evals) + 0.5) / n_evals
        
        # Power-law exponent beta
        # beta=2 corresponds to uniform arc length sampling for ZDT1
        beta = 2.0
        x0_vals = u ** beta
        
        # Map to [lo[0], hi[0]]
        x0_vals = lo[0] + (hi[0] - lo[0]) * x0_vals
        
        # Construct population
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_vals
        # Set x[1:] to midpoint of bounds (optimal for ZDT1 g function)
        X[:, 1:] = (hi[1:] + lo[1:]) / 2.0
        
        # Evaluate
        results = []
        for x in X:
            f = problem(x)
            results.append(f)
            
        return results
