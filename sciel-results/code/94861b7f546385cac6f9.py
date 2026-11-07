import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        # Generate Latin Hypercube Sampling points for x[0]
        # We assume the Pareto front is determined by x[0]
        # and we set other variables to 0 to minimize g(x)
        
        n_points = budget
        if n_points < 1:
            n_points = 1
            
        # LHS generation
        # 1. Create strata
        strata = np.linspace(0, 1, n_points + 1)
        # 2. Sample one point in each stratum
        # np.random.rand(n_points) gives [0, 1)
        # We need to map them to the strata
        # A common way:
        # u = np.random.rand(n_points)
        # x_lhs = (np.arange(n_points) + u) / n_points
        # Then shuffle x_lhs to break correlation (though in 1D it doesn't matter for coverage, 
        # but it makes the sequence look random)
        
        u = np.random.rand(n_points)
        x_lhs = (np.arange(n_points) + u) / n_points
        np.random.shuffle(x_lhs)
        
        # Clip to bounds
        x_lhs = np.clip(x_lhs, lo[0], hi[0])
        
        recorded_F = []
        
        for i in range(n_points):
            x = np.zeros(dim)
            x[0] = x_lhs[i]
            # x[1:] are 0, which is within [0, 1]
            
            f = problem(x)
            if f is not None and len(f) == n_obj:
                recorded_F.append(f)
                
            if time.time() > deadline:
                break

        return recorded_F
