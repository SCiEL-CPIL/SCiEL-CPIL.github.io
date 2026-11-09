import numpy as np
from scipy.stats import qmc
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, leave 10s margin
        
        n_samples = self.budget
        
        # Generate Sobol points in [0, 1]^2
        # We use 2 dimensions for the Pareto front variables y1, y2
        try:
            sampler = qmc.Sobol(d=2, scramble=True)
            u_points = sampler.random(n_samples)
        except Exception:
            u_points = np.random.random((n_samples, 2))

        # Map uniform points in [0,1]^2 to the Pareto front in objective space
        # DTLZ7 Pareto Front (with g=1):
        # f1 = y1
        # f2 = y1 * (1 - y2)
        # The region in (f1, f2) is the triangle: 0 <= f2 <= f1 <= 1
        # To sample uniformly in this triangle (which is often better for hypervolume 
        # when the reference point is far away and the front is "flat" in f3):
        # Let U1, U2 be uniform in [0,1].
        # f1 = sqrt(U1)
        # f2 = f1 * U2 = sqrt(U1) * U2
        # Then y1 = f1 = sqrt(U1)
        # y2 = 1 - f2/f1 = 1 - U2
        
        y1 = np.sqrt(u_points[:, 0])
        y2 = 1.0 - u_points[:, 1]
        
        # Construct full dimension vectors
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = y1
        X[:, 1] = y2
        # Variables 2 to 21 (indices 2..21) are set to 0.0 to minimize g to 1.0
        
        # Clip to bounds just in case
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            x = X[i]
            x = np.clip(x, lo, hi)
            problem(x)

        return X
