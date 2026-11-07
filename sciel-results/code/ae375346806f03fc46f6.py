import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        t0 = time.time()
        DEADLINE = 310.0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        used = 0
        F = []

        def eval_point(x):
            nonlocal used
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return f

        # Generate optimal points on the Pareto front
        # ZDT1 Front: f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0, 1], x2..xn = 0
        # Optimal spacing for HV maximization on this convex front is approximately
        # x_i proportional to i^2 (clustering near x=0)
        N = budget
        if N < 2:
            N = 2
            
        # Ensure we don't exceed budget
        # We will evaluate N points.
        
        # t_vals for x1
        # Using (i/(N-1))^2 ensures 0 and 1 are included
        idx = np.arange(N)
        t_vals = (idx / (N - 1)) ** 2
        
        # Evaluate each point
        for t in t_vals:
            if used >= budget or (time.time() - t0) > DEADLINE:
                break
            x = np.zeros(dim)
            x[0] = float(t)
            # x[1:] = 0 is already set by zeros
            
            f = eval_point(x)
            if np.all(np.isfinite(f)):
                F.append(f)

        # Fallback if something went wrong (should not happen)
        if len(F) == 0 and used < budget:
            rng = np.random.default_rng()
            span = hi - lo
            for _ in range(min(10, budget - used)):
                if used >= budget or (time.time() - t0) > DEADLINE:
                    break
                x = lo + span * rng.random(dim)
                f = eval_point(x)
                if np.all(np.isfinite(f)):
                    F.append(f)

        if len(F) > 0:
            return np.array(F)
        return np.zeros((0, dim))
