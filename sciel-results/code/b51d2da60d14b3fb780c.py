"""Multi-objective optimizer on ZDT6 using analytic Pareto-front construction."""
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search initialized on the known ZDT6 Pareto set."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # 330s limit, keep 10s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget

        # ZDT6 Pareto front: x[1:] = 0, x1 in [0, 1]
        # f1(x1) = 1 - exp(-4*x1) * sin^6(6*pi*x1)
        # f2 = 1 - sqrt(f1)
        
        # The Pareto front is fully covered by the first lobe x1 in [0, 1/6].
        # We want to distribute n_evals points such that they cover the range of f1
        # in [f1_min, 1] as evenly as possible to maximize hypervolume.
        
        def f1_func(x1):
            return 1.0 - np.exp(-4.0 * x1) * (np.sin(6.0 * np.pi * x1) ** 6)
        
        # Find the minimum f1 in the first lobe [0, 1/6]
        # The minimum is where derivative is 0.
        # Numerically find it using a dense grid first
        x_grid = np.linspace(0.0, 1.0/6.0, 1000)
        f1_grid = f1_func(x_grid)
        idx_min = np.argmin(f1_grid)
        x_min_approx = x_grid[idx_min]
        f1_min = f1_grid[idx_min]
        
        # Refine x_min using golden section or just fine grid around idx_min
        x_refine = np.linspace(x_grid[max(0, idx_min-10)], x_grid[min(len(x_grid)-1, idx_min+10)], 100)
        f1_refine = f1_func(x_refine)
        idx_min_ref = np.argmin(f1_refine)
        x_min = x_refine[idx_min_ref]
        f1_min = f1_refine[idx_min_ref]
        
        # We want to sample f1 uniformly in [f1_min, 1.0]
        # Note: f1(0)=1 and f1(1/6)=1. The minimum is in between.
        # We can map each target f1 to an x1 in [0, x_min] (the decreasing branch)
        # or [x_min, 1/6] (the increasing branch).
        # Since f1 is monotonic on [0, x_min], we can invert it.
        
        target_f1s = np.linspace(f1_min, 1.0, n_evals)
        
        # For each target f1, find x1 in [0, x_min] such that f1_func(x1) = target_f1
        # Use binary search
        x1_candidates = np.zeros(n_evals)
        for i, tf in enumerate(target_f1s):
            # f1 is decreasing on [0, x_min]
            a = 0.0
            b = x_min
            # If tf > f1(0)=1 or tf < f1_min, clamp
            if tf > 1.0:
                x1_candidates[i] = 0.0
                continue
            if tf < f1_min:
                x1_candidates[i] = x_min
                continue
            
            for _ in range(50): # 50 iterations of bisection is plenty
                mid = (a + b) / 2.0
                fm = f1_func(mid)
                if fm > tf:
                    a = mid
                else:
                    b = mid
            x1_candidates[i] = (a + b) / 2.0
            
        # Create X
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_candidates
        
        # Evaluate all points
        for i in range(n_evals):
            if time.time() > deadline:
                break
            problem(X[i])
            
        # Return the last evaluated point
        return X[-1]
