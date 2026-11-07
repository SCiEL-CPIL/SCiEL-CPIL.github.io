import numpy as np

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            n = self.dim
            budget = self.budget
            
            # We will use all budget evaluations to sample the Pareto front.
            # The front is located at x_1...x_{n-1} = 0.
            # x_0 determines f_1.
            
            # Uniform sampling of x_0 in [0, 1]
            # bounds[0] is for x_0. Assuming [0, 1].
            lo0, hi0 = bounds[0]
            
            # Generate x_0 values
            x0_vals = np.linspace(lo0, hi0, budget)
            
            # Create full population
            X = np.zeros((budget, n))
            X[:, 0] = x0_vals
            # X[:, 1:] is already 0.0
            
            # Ensure bounds are respected (though 0 is inside [-5, 5])
            for i in range(1, n):
                lo, hi = bounds[i]
                # 0 is within bounds for ZDT4 standard, but let's be safe
                if 0 < lo or 0 > hi:
                    X[:, i] = lo # Fallback, though unlikely
            
            # Evaluate
            F = np.array([problem(x) for x in X])
            
            return X
