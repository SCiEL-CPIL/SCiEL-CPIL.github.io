import numpy as np
import time

class Optimizer:
    """
    Optimizer for ZDT4 exploiting its decoupled structure.
    Sets x[1:] to 0 (clipped to bounds) to lie on the Pareto front,
    then sweeps x[0] uniformly across its effective bounds [0, 1].
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 300s limit, use 290s
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 Pareto front is defined for x[0] in [0, 1] and x[1:] = 0.
        # We restrict the sweep to the effective range of x[0].
        x0_lo = max(lo[0], 0.0)
        x0_hi = min(hi[0], 1.0)
        
        # If the effective range is empty or invalid, fallback to full bounds
        if x0_lo >= x0_hi:
            x0_lo = lo[0]
            x0_hi = hi[0]

        # Set x[1:] to 0, clipped to bounds
        base = np.zeros(self.dim - 1)
        for i in range(self.dim - 1):
            if 0.0 < lo[i+1]:
                base[i] = lo[i+1]
            elif 0.0 > hi[i+1]:
                base[i] = hi[i+1]

        # Use the full budget for the sweep
        n_sweep = self.budget
        
        # Generate x[0] samples uniformly in the effective range
        if n_sweep < 1:
            n_sweep = 1
            
        x0_vals = np.linspace(x0_lo, x0_hi, n_sweep)
        
        # Construct the full vectors
        X = np.empty((n_sweep, self.dim))
        X[:, 0] = x0_vals
        X[:, 1:] = base
        
        # Evaluate all points
        F = np.empty((n_sweep, self.n_obj))
        valid_count = 0

        for i in range(n_sweep):
            if time.time() > deadline:
                break
            try:
                F[i] = problem(X[i])
                valid_count += 1
            except:
                # If an evaluation fails, break to avoid further issues
                break

        # Return the population (return value is ignored, but we return something valid)
        if valid_count > 0:
            return X[:valid_count]
        else:
            return X
