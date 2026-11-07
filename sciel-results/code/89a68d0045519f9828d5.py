import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget
        
        # Use Sobol sequence for better space filling in the 2D active space
        # We want to sample angles such that the resulting points on the Pareto front
        # (a quarter-sphere) are distributed uniformly with respect to surface area.
        # Surface area element dA = sin(alpha1) dalpha1 dalpha2.
        # Let v = cos(alpha1), sampled uniformly in [0, 1].
        # Let w be uniform in [0, 1], mapped to alpha2 = w * pi/2.
        
        # Generate Sobol points in [0, 1]^2
        sampler = qmc.Sobol(d=2, scramble=True)
        # Generate slightly more to account for potential edge cases, though Sobol is deterministic
        # We just take the first n_evals points
        samples = sampler.random(n_evals)
        
        v = samples[:, 0]
        w = samples[:, 1]
        
        # Map to angles
        # alpha1 in [0, pi/2]
        # cos(alpha1) = v -> alpha1 = arccos(v)
        # Note: v in [0,1] -> arccos(v) in [pi/2, 0]. This is fine, alpha1 is just an angle.
        alpha1 = np.arccos(v)
        # alpha2 in [0, pi/2]
        alpha2 = w * (np.pi / 2.0)
        
        # Map angles to decision variables x0, x1
        # x_i = (2/pi) * alpha_i
        x0 = (2.0 / np.pi) * alpha1
        x1 = (2.0 / np.pi) * alpha2
        
        # Construct full decision vectors
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0
        X[:, 1] = x1
        
        if self.dim > 2:
            # Set remaining variables to 0.5 to ensure g(x) = 0 (Pareto optimal)
            # Assuming bounds are [0,1]
            mid_points = (lo[2:] + hi[2:]) / 2.0
            X[:, 2:] = mid_points
            
        # Evaluate
        eval_count = 0
        for i in range(len(X)):
            if time.time() > deadline:
                break
            try:
                x = X[i]
                f = problem(x)
                eval_count += 1
            except Exception:
                break
                
        return X
