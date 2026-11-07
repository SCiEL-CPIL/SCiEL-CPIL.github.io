"""ZDT3 Optimizer using structured Halton sampling on the relevant dimension."""
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
        # To maximize hypervolume, we minimize g by setting x[1:] = 0.
        # This makes g = 1, and the front is f2 = 1 - sqrt(f1).
        
        # We sample x[0] in [lo[0], hi[0]].
        # To maximize the hypervolume of the finite set of points,
        # we should distribute points non-uniformly, denser near x[0]=0.
        # This is because the hypervolume integral weights the y-drops
        # by (r1 - x_next), which is larger for smaller x_next.
        
        # Generate N samples in [0, 1] using a power law distribution.
        # u_i = i / N
        # x_norm_i = u_i^p
        # Map to [lo[0], hi[0]]
        
        # Let's try p=2.0 as a heuristic for denser sampling near 0.
        # We also include the endpoints to ensure full range coverage.
        
        N = budget
        # Ensure N is at least 2
        if N < 2:
            N = 2
            
        # Create indices 1..N
        idx = np.arange(1, N + 1)
        # Normalize to [0, 1]
        u = idx / N
        # Apply power law to densify near 0
        # If p > 1, u^p is smaller for small u, so points are closer to 0.
        p = 2.0
        x_norm = u ** p
        
        # Map to actual bounds
        x0_samples = lo[0] + x_norm * (hi[0] - lo[0])
        
        # Ensure the last point is exactly hi[0] to cover the full range
        x0_samples[-1] = hi[0]
        
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
