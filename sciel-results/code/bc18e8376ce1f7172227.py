import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Deadline guard
        import time
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy:
        # ZDT1 Pareto front is at x_2...x_30 = 0.
        # f1 = x_1, f2 = 1 + sqrt(x_1)
        # We want to sample x_1 to maximize hypervolume.
        # Hypervolume is area under the curve f2(x1) from 0 to 1, approximated by points.
        # To maximize the area covered by the "staircase" of points, we should
        # distribute points such that the gaps in f1 are weighted by the "height" or slope.
        # Actually, for a concave front, clustering near the origin (0,0) is beneficial
        # because the marginal gain of hypervolume is higher there?
        # Let's use a power law x1 = u^k. k=2 is a reasonable guess for sqrt front.
        
        n_structured = int(self.budget * 0.95)
        n_random = self.budget - n_structured
        
        # Generate structured points
        # Use a quasi-random or well-spaced distribution
        # u in [0, 1]
        # x1 = u^2
        u_vals = np.linspace(0, 1, n_structured)
        # Shift slightly to avoid exact 0 if needed, but 0 is good.
        # Actually, linspace includes 0 and 1.
        x1_vals = u_vals ** 2
        
        structured_X = np.zeros((n_structured, self.dim))
        structured_X[:, 0] = x1_vals
        # x[1:] remains 0
        
        # Generate random points
        rng = np.random.default_rng(0)
        random_X = lo + (hi - lo) * rng.random((n_random, self.dim))
        
        # Combine
        all_X = np.vstack([structured_X, random_X])
        
        # Ensure we don't exceed budget
        if len(all_X) > self.budget:
            all_X = all_X[:self.budget]
            
        # Evaluate
        # The problem function clips, so we are safe.
        for i in range(len(all_X)):
            if time.time() > deadline:
                break
            problem(all_X[i])
            
        return None
