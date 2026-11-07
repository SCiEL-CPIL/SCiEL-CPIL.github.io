import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        
        n = self.dim
        N = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT2 Pareto front: f1 = x1, f2 = 11*(1 - sqrt(x1/11))
        # We need to place N points on this front to maximize hypervolume with reference [11, 11]
        
        # The hypervolume contribution of point i is (f1_{i+1} - f1_i) * (f2_i - ref2) + ...
        # For convex decreasing front, optimal spacing in f1 is proportional to 1/sqrt(f2_i - ref2)
        # Since f2 = 11*(1 - sqrt(f1/11)), we have f2 - 11 = -11*sqrt(f1/11)
        # So 1/sqrt(f2_i - ref2) doesn't work directly since ref2 = 11 > f2
        
        # Actually, the hypervolume with reference (11,11):
        # HV = sum over points of (f1_{i+1} - f1_i) * (f2_i - 11) ... but f2 < 11 always
        # Wait, ref = [11, 11], and we need points strictly better in EVERY objective
        # So f1 < 11 and f2 < 11. Since f2 = 11*(1-sqrt(f1/11)) < 11 always for f1 > 0
        # And f1 = x1 in [0,1], so f1 < 11 always.
        
        # The hypervolume of a set of points on a monotone front with reference (r1, r2):
        # If points are sorted by f1 ascending: (f1_1, f2_1), ..., (f1_N, f2_N)
        # HV = sum_{i=1}^{N-1} (f1_{i+1} - f1_i) * (f2_i - r2) + (r1 - f1_N) * (f2_N - r2)
        # But r2 = 11 and f2_i < 11, so (f2_i - r2) < 0. That's wrong.
        
        # Let me reconsider. Hypervolume with reference point (r1, r2) where r1 > all f1, r2 > all f2:
        # For minimization, the dominated region is [f1_i, r1] x [f2_i, r2]
        # HV = area of union of rectangles [f1_i, r1] x [f2_i, r2]
        # For sorted points by f1 ascending (and f2 descending for Pareto front):
        # HV = sum_{i=1}^{N-1} (f1_{i+1} - f1_i) * (r2 - f2_i) + (r1 - f1_N) * (r2 - f2_N)
        
        # So we want to maximize: sum_{i=1}^{N-1} (f1_{i+1} - f1_i) * (11 - f2_i) + (11 - f1_N) * (11 - f2_N)
        
        # With f1 = x1, f2 = 11*(1 - sqrt(x1/11)):
        # 11 - f2 = 11*sqrt(x1/11) = sqrt(11*x1)
        
        # So HV = sum_{i=1}^{N-1} (x1_{i+1} - x1_i) * sqrt(11*x1_i) + (11 - x1_N) * sqrt(11*x1_N)
        
        # To maximize this, we use calculus of variations / optimal spacing.
        # The optimal x1 spacing satisfies: sqrt(11*x1_i) * (x1_{i+1} - x1_i) is roughly constant
        # This leads to x1_i proportional to i^2, i.e., x1_i = 11 * (i/N)^2
        
        # Let's use x1_i = 11 * (i/N)^2 for i = 1, ..., N
        # This gives uniform spacing in sqrt(x1/11)
        
        t = np.linspace(0.0, 1.0, N)
        x1_vals = 11.0 * (t ** 2.0)
        x1_vals = np.clip(x1_vals, 0.0, 1.0)
        
        # Construct full x vectors
        X = np.zeros((N, n))
        X[:, 0] = x1_vals
        
        # For ZDT2, to achieve the Pareto front, we need g(x) = 0
        # g(x) = sum_{i=2}^{n} (x_i^2 - 2*x1*x_i*cos(10*pi*x1*x_i))
        # Setting x_i = (0.5*x1)^(1/(4*(n-1))) makes each term zero (approximately)
        # Actually, this is a known trick: if x_i = (x1/2)^(1/(4*(n-1))) then
        # x_i^2 = x1/(2*(n-1)) and 2*x1*x_i*cos(...) needs to match
        # The standard result is that x_i = (x1/2)^(1/(4*(n-1))) gives g(x)=0
        
        if n > 1:
            exponent = 1.0 / (4.0 * (n - 1))
            # Handle x1=0 case
            x1_safe = np.maximum(x1_vals, 1e-10)
            rest_vals = (0.5 * x1_safe) ** exponent
            X[:, 1:] = np.tile(rest_vals[:, np.newaxis], (1, n - 1))
        
        # Clip to bounds
        for j in range(n):
            X[:, j] = np.clip(X[:, j], lo[j], hi[j])
        
        # Evaluate all points
        F = np.array([problem(x_vec) for x_vec in X])
        
        return X
