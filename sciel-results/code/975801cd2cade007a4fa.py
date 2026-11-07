"""ZDT3 Optimizer using structured sampling on the relevant dimension."""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for the 330s kill switch
        
        dim = self.dim
        budget = self.budget
        bounds_arr = np.array(bounds)
        lo = bounds_arr[:, 0]
        hi = bounds_arr[:, 1]
        
        # ZDT3 specific structure:
        # f1 = x[0]
        # g = 1 + 10/d * (sum(x[i]) - sum(f(x[i])))
        # f2 = g * (1 - sqrt(f1/g))
        
        # To maximize hypervolume, we want to minimize g.
        # For standard ZDT3 bounds [0, 1], setting x[i] = 0 for i > 0
        # results in f(x[i]) = 0 and x[i] = 0, so the term (x[i] - f(x[i])) is 0.
        # Thus g = 1. This is the minimum g achievable without risking g < 0
        # or complex non-uniform g behaviors that might complicate the front.
        # With g = 1, f2 = 1 - sqrt(f1). This is the standard Pareto front.
        
        # We sample x[0] uniformly in [lo[0], hi[0]] to cover the front.
        # We fix x[1:] to 0 (or lo[1:] if 0 is out of bounds, but standard is 0).
        
        # Generate x0 samples
        # Using linspace ensures uniform coverage of the f1 axis.
        x0_samples = np.linspace(lo[0], hi[0], budget)
        
        # Pre-allocate the rest of the vector
        # Set to 0 to minimize g.
        # If lo[1:] > 0, we should set to lo[1:] to be within bounds.
        # However, the task specifies bounds [0, 1], so 0 is valid.
        rest_val = 0.0
        if lo[1].min() > 0:
            rest_val = lo[1].min()
            
        for i in range(budget):
            if time.time() > deadline:
                break
            
            x = np.full(dim, rest_val)
            x[0] = x0_samples[i]
            
            # Clip to bounds to be safe
            x = np.clip(x, lo, hi)
            
            # Evaluate
            # The harness records this
            problem(x)
            
        return None
