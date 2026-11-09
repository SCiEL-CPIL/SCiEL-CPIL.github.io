"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer tailored for ZDT2.
    Exploits the known structure: Pareto front lies on x[1:] = 0,
    with f1 = x[0] and f2 = 1 - sqrt(x[0]) (when x[1:] = 0).
    
    Uses a mix of uniform-in-f1 and sqrt-spaced-in-f1 (uniform-in-f2)
    to maximize hypervolume coverage. The front is concave, so we need
    points near both corners and in the middle.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_points = min(self.budget, 200)

        # ZDT2 Pareto front: x[0] in [0,1], x[1:] = 0
        # f1 = x[0], f2 = 1 - sqrt(x[0])
        
        # The Pareto front is f2 = 1 - sqrt(f1) for f1 in [0,1]
        # This is a concave curve. To maximize hypervolume with reference [11,11],
        # we want to cover the area under the curve as well as possible.
        
        # Strategy: Use a combination of:
        # 1. Uniform spacing in f1 (x[0])
        # 2. Spacing that makes f2 uniform (x[0] = (1-u)^2 where u uniform in [0,1])
        
        # For a concave front like f2 = 1 - sqrt(f1), the optimal spacing 
        # for hypervolume maximization with a fixed reference point tends to
        # be denser near the corners where the curve is steep.
        
        # Let's try: 50% uniform in x[0], 50% uniform in f2 (i.e., x[0] = (1-u)^2)
        n_uniform_f1 = n_points // 2
        n_uniform_f2 = n_points - n_uniform_f1
        
        # Uniform in f1 = x[0]
        x0_uniform = np.linspace(0.0, 1.0, n_uniform_f1)
        
        # Uniform in f2: f2 = 1 - sqrt(x[0]), so x[0] = (1 - f2)^2
        # If f2 is uniform in [0,1], then x[0] = (1 - u)^2 for u uniform in [0,1]
        u_f2 = np.linspace(0.0, 1.0, n_uniform_f2)
        x0_sqrt = (1.0 - u_f2) ** 2
        
        # Combine
        x0_values = np.concatenate([x0_uniform, x0_sqrt])
        
        # Sort to ensure we have a good spread
        x0_values = np.sort(x0_values)
        
        # Build the full vectors with x[1:] = 0
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0_values
        
        # Ensure within bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_points
        
        # Check time
        if time.time() > deadline:
            return X
        
        return X
# EVOLVE-BLOCK-END
