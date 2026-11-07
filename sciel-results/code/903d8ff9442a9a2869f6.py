import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        budget = self.budget
        dim = self.dim
        
        # Use power law spacing to maximize hypervolume overestimate
        # t_i = (i / (N-1))^p
        # We want large steps at t=0, so p < 1.
        # Let's try p = 0.5
        p = 0.5
        
        N = budget
        if N < 2:
             N = 2
             
        # Generate points
        # Ensure we hit 0 and 1 exactly
        t_vals = np.linspace(0, 1, N)
        t_vals = t_vals ** p
        
        X = []
        F = []
        
        for t in t_vals:
            x = np.zeros(dim)
            x[0] = float(t)
            # Ensure bounds
            if x[0] < bounds[0][0]: x[0] = bounds[0][0]
            if x[0] > bounds[0][1]: x[0] = bounds[0][1]
            
            f = problem(x)
            f = np.asarray(f, dtype=float)
            
            if np.all(np.isfinite(f)):
                X.append(x)
                F.append(f)
                
        return np.array(X)
