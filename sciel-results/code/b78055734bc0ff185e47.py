# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Prototype-based sweep optimizer for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        all_x = []
        all_f = []
        used = 0

        def eval_point(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_x.append(x)
            all_f.append(f)
            used += 1
            return f

        # Step 1: Determine the best base configuration for x[1:]
        # Test a few candidates at x[0] = 0.5
        test_x0 = 0.5
        candidates = [
            np.full(dim, lo[1]),       # Lower bounds
            np.full(dim, hi[1]),       # Upper bounds
            np.full(dim, 0.5),         # Center
            lo[1:] + (hi[1:] - lo[1:]) * np.random.random(dim - 1) # Random
        ]
        
        best_candidate = None
        best_f2 = float('inf')
        
        for c in candidates:
            if used >= budget:
                break
            x_test = np.empty(dim)
            x_test[0] = test_x0
            x_test[1:] = c
            f_test = eval_point(x_test)
            if f_test[1] < best_f2:
                best_f2 = f_test[1]
                best_candidate = c.copy()
                
        if best_candidate is None:
            best_candidate = np.full(dim - 1, 0.5) # Fallback

        # Step 2: Sweep x[0] using the best candidate for x[1:]
        # We have spent ~4 evaluations. Remaining: budget - 4.
        n_sweep = budget - used - 5 # Leave a small buffer
        if n_sweep < 10:
            n_sweep = max(10, budget - used)
            
        x0_values = np.linspace(0.0, 1.0, n_sweep)
        
        for x0 in x0_values:
            if used >= budget or time.time() > deadline:
                break
            x_sweep = np.empty(dim)
            x_sweep[0] = x0
            x_sweep[1:] = best_candidate
            eval_point(x_sweep)
            
        # Step 3: If budget remains, do random search
        while used < budget and time.time() < deadline:
            x = lo + (hi - lo) * np.random.random(dim)
            eval_point(x)

        return all_x
# EVOLVE-BLOCK-END
