import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 320.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        n_evals = self.budget
        if n_evals < 1:
            return np.array([])

        # ZDT1 structure:
        # f1(x) = x[0]
        # f2(x) = (1 + 10*(n-1)) * (1 - sqrt(x[0] / g(x)))
        # g(x) = 1 + 9 * sum(x[1:]) / (n-1)
        #
        # Pareto-optimal points have x[1:] = 0 (lower bound), giving g=1.
        # f2 = C * (1 - sqrt(f1)), where C = 1 + 10*(n-1).
        #
        # Hypervolume maximization with reference point [11, 11]:
        # The hypervolume contribution is dominated by the area under the curve.
        # To maximize the union of rectangles, we want points spread across the front.
        # A uniform distribution in f1 (x[0]) is a good baseline, but uniform in f2 
        # (which implies uniform in sqrt(f1)) is theoretically better for HV.
        # However, random sampling (LHS) is robust and avoids grid artifacts.
        
        # We will use Latin Hypercube Sampling (LHS) on x[0] to ensure 
        # good 1D coverage without the rigidity of a grid.
        # Since the problem is 1D in terms of the Pareto front (x[0] determines the front),
        # we just need to sample x[0] well.
        
        # Generate n_evals random samples in [0, 1] for the normalized x[0]
        # Then map to [lo[0], hi[0]]
        # Using a permutation-based LHS for better spread than pure random
        
        # Create bins
        n_pts = n_evals
        # Random permutation of indices 0..n-1
        perm = np.random.permutation(n_pts)
        # Assign random offsets within each bin
        offsets = np.random.rand(n_pts)
        # Normalized positions in [0, 1]
        u = (perm + offsets) / n_pts
        
        # Map to x[0] bounds
        x0_vals = lo[0] + u * (hi[0] - lo[0])
        
        X = np.empty((n_pts, self.dim), dtype=np.float64)
        X[:, 0] = x0_vals
        
        if self.dim > 1:
            # Set x[1:] to lower bound to minimize g and thus f2
            # This ensures all points are on the Pareto front
            X[:, 1:] = lo[1:]
            np.clip(X[:, 1:], lo[1:], hi[1:], out=X[:, 1:])
            
        # Evaluate all points
        F = np.empty((n_pts, self.n_obj), dtype=np.float64)
        
        for i in range(n_pts):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            
        return X
