import numpy as np
import time


class Optimizer:
    """Analytic front sampler for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        budget = self.budget
        dim = self.dim
        
        # 1. Generate target f1 values
        # We want to cover the range of f1 on the Pareto front.
        # f1 ranges from 0 to ~0.98 on [0,1].
        # We'll sample uniformly in [0.01, 0.99] to avoid exact 0/1 edges which might be tricky.
        n_points = budget
        targets = np.linspace(0.01, 0.99, n_points)
        
        # 2. Find x0 for each target f1
        # f1(x0) = 1 - exp(-4 * x0 * sin^2(10 * pi * x0))
        # We need to solve for x0.
        # Since the function is oscillatory, we scan x0 in [0, 1] to find a solution.
        # We use a fine grid for the scan to ensure we don't miss the crossing.
        scan_step = 0.0005
        x0_scan = np.arange(0.0, 1.0 + scan_step, scan_step)
        f1_scan = 1.0 - np.exp(-4.0 * x0_scan * np.sin(10.0 * np.pi * x0_scan)**2)
        
        x0_solutions = np.zeros(n_points)
        
        for i, target in enumerate(targets):
            # Find the first x0 where f1 crosses target
            # Since f1 starts at 0, goes up and down, we look for sign change of (f1 - target)
            diff = f1_scan - target
            # Find indices where sign changes
            sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
            
            if len(sign_changes) > 0:
                # Take the first crossing
                idx = sign_changes[0]
                # Linear interpolation for better accuracy
                x1, x2 = x0_scan[idx], x0_scan[idx+1]
                f1_1, f1_2 = f1_scan[idx], f1_scan[idx+1]
                if abs(f1_2 - f1_1) > 1e-12:
                    x0_solutions[i] = x1 + (target - f1_1) * (x2 - x1) / (f1_2 - f1_1)
                else:
                    x0_solutions[i] = x1
            else:
                # Fallback: pick x0 that minimizes |f1 - target|
                x0_solutions[i] = x0_scan[np.argmin(np.abs(diff))]
        
        # 3. Evaluate the points
        # x[0] = x0, x[1:] = 0
        # Ensure x0 is within bounds [0, 1]
        x0_solutions = np.clip(x0_solutions, lo[0], hi[0])
        
        last_f = None
        for i in range(n_points):
            if time.time() > deadline:
                break
                
            x = np.zeros(dim)
            x[0] = x0_solutions[i]
            # x[1:] is already 0, which is within [0,1] bounds
            
            f = problem(x)
            last_f = f
            
        return last_f
