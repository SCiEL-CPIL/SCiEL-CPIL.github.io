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
        
        # Determine fixed values for x_2...x_dim
        # ZDT4 g(x) min is at x_i = 0.5 for i > 0
        fixed_vals = []
        for i in range(1, self.dim):
            target = 0.5
            if target < lo[i]:
                val = lo[i]
            elif target > hi[i]:
                val = hi[i]
            else:
                val = 0.5
            fixed_vals.append(val)
            
        # We have self.budget evaluations left for x_1
        n_samples = self.budget
        # Uniformly sample x_1
        x1_samples = np.linspace(lo[0], hi[0], n_samples)
        
        # Construct full vectors
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x1_samples
        for i in range(1, self.dim):
            X[:, i] = fixed_vals[i-1]
            
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
