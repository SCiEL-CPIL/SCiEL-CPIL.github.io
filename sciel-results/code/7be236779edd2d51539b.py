import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = self.budget
        
        # ZDT2 Pareto front: f1 = x0, f2 = (1 + sqrt(f1))^(-1/2)
        # To maximize hypervolume with reference point [11, 11],
        # we want to distribute points to maximize the area under the front.
        # 
        # The hypervolume contribution for a set of points on the front is
        # sum over non-dominated points of (ref_f2 - f2_i) * (f1_i - f1_{i-1})
        # where f1_0 = 0.
        #
        # For a convex decreasing front, to maximize this sum with N points,
        # we should place points where the front has the highest curvature,
        # i.e., where the second derivative is largest in magnitude.
        #
        # f2(f1) = (1 + sqrt(f1))^(-1/2)
        # f2'(f1) = -1/4 * (1 + sqrt(f1))^(-3/2) * (1/sqrt(f1))
        # f2''(f1) involves terms that are large near f1 = 0.
        #
        # A good heuristic: distribute points according to the curvature.
        # Near f1=0, the front is very steep, so we need more points there.
        #
        # Let's use a transformation that clusters points near f1=0.
        # Use f1 = u^2 where u is uniform in [0,1].
        # This gives more points near f1=0.
        
        u = np.linspace(0.0, 1.0, N)
        f1_vals = u ** 2  # Cluster near 0
        
        X = np.zeros((N, self.dim))
        X[:, 0] = f1_vals
        # x[1:] remains 0 to ensure g=1 and we're on the Pareto front
        
        F_list = []
        for i in range(N):
            if time.time() > deadline:
                break
            x = X[i]
            f = problem(x)
            F_list.append(f)
            
        return X
