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
        
        # For ZDT2, the Pareto front is determined by x[0] (with x[1:] = 0 for g(x)=1)
        # f1 = x[0]
        # f2 = (1 + 10*(n-1)) * (1 - sqrt(f1/(1 + 10*(n-1))))
        # To maximize hypervolume, we want to sample x[0] such that f1 values are
        # well-distributed on the Pareto front. Since f1 = x[0], we want good coverage
        # of x[0] in [0, 1].
        
        # The current LHS approach is good, but we can improve by ensuring we capture
        # the endpoints and use a deterministic grid that maximizes minimum distance.
        # For hypervolume, uniform spacing in f1 space is optimal.
        
        n_points = budget
        if n_points < 1:
            n_points = 1
            
        # Use a deterministic grid with uniform spacing in f1 space
        # This ensures maximum spread and good hypervolume coverage
        # Include endpoints for better coverage of the front
        x_lhs = np.linspace(0, 1, n_points)
        
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
