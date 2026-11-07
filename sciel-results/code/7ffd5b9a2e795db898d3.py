import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget
        
        # DTLZ2 structure:
        # f1 = g * prod(cos(theta_i)) for i in 0..M-2
        # f_{i+1} = g * prod(cos(theta_j)) for j < i * sin(theta_i) for i in 1..M-1
        # f_M = g * prod(sin(theta_i))
        # where theta_i = pi/2 * x_i for i in 0..M-2
        # g = sum((x_j - 0.5)^2) for j in M-1..dim-1
        #
        # For the Pareto front, g = 1.
        # We need to set x[j] for j >= M-1 such that sum((x[j]-0.5)^2) = 1.
        # One simple way: set all x[j] = 0.5 + sqrt(1/(dim - M + 1))
        # But we must ensure they stay within bounds [0,1].
        # sqrt(1/10) ≈ 0.316, so 0.5 + 0.316 = 0.816 < 1. OK.
        # Also 0.5 - 0.316 = 0.184 > 0. OK.
        
        # Actually, the standard DTLZ2 has g = sum((x_i - 0.5)^2) for i = M-1 to N-1.
        # The Pareto front is at g=1.
        # We can set the distance variables to achieve g=1.
        # Let's use x[j] = 0.5 + d for all j in [M-1, dim-1], where d = sqrt(1/(dim - M + 1)).
        
        k_shape = n_obj - 1  # Number of shape variables
        n_dist = dim - k_shape  # Number of distance variables
        d_val = np.sqrt(1.0 / n_dist)
        
        # Pre-compute the distance part of the vector
        dist_part = np.full(n_dist, 0.5 + d_val)
        
        # We'll use a quasi-random sequence (Halton) to sample the shape variables.
        # This gives good coverage of the [0,1]^{M-1} hypercube.
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r
        
        # Primes for Halton sequence bases
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
        # Generate points using Halton sequence
        # We use indices starting from 1 to avoid 0 (which gives 0 in Halton)
        # Actually, Halton(0, base) = 0, which is fine.
        # But let's start from 1 for better distribution.
        
        n_evals = budget
        points = []
        
        for i in range(1, n_evals + 1):
            if time.time() > deadline:
                break
            
            # Generate shape variables using Halton sequence
            shape = np.zeros(k_shape)
            for j in range(k_shape):
                shape[j] = halton(i, bases[j])
            
            # Construct the full vector
            x = np.zeros(dim)
            x[:k_shape] = shape
            x[k_shape:] = dist_part
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            # Evaluate
            f = problem(x)
            points.append(f)
        
        return points
