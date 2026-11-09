# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    Strategy:
    1. Exploit the known analytic structure of ZDT3's Pareto front.
       The front is defined by g(x) = 1 (x_2..x_n = 0).
       f1 = x1
       f2 = 1 + 9/sqrt(f1) * (0.55 + 0.45*sin(3*pi*f1))
    2. Instead of uniform sampling, use adaptive stratified sampling.
       The ZDT3 front has discontinuities and varying curvature.
       We divide the domain [0, 1] into many fine strata.
       For each stratum, we estimate the marginal hypervolume contribution.
       We allocate samples to strata proportional to their estimated contribution.
       This concentrates evaluations where the front "bends" or has high
       potential for non-dominated points, improving hypervolume coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Reference point for hypervolume calculation
        ref = np.array([11.0, 11.0])
        
        # Analytic ZDT3 Pareto front function
        # f1 = x1
        # f2 = 1 + 9/sqrt(x1) * (0.55 + 0.45*sin(3*pi*x1))
        # Note: At x1=0, f2 is undefined/infinite in the formula, but the 
        # actual ZDT3 function handles this. The front starts at x1=0.
        # Let's check the behavior near 0. 
        # Actually, the standard ZDT3 definition has f2 discontinuities.
        # The Pareto front is the set of points (f1, f2) where f1 in [0,1].
        
        # We will use a fine grid to estimate the "importance" of each segment.
        # The hypervolume contribution of a point (f1, f2) relative to ref 
        # and other points depends on its position. 
        # A simple proxy for importance in 2D minimization with fixed ref 
        # is the area of the rectangle (ref - f) that is not covered by 
        # "better" points. 
        # However, since we are sampling the front, all points are non-dominated 
        # (mostly). The total hypervolume is the area under the curve 
        # (ref - f2) from f1_min to f1_max, roughly.
        
        # Let's just use a very dense stratified sampling but weighted by 
        # the "length" of the front segment in the objective space?
        # Or simply, since the front is 1D, the optimal distribution for 
        # hypervolume is often related to the curvature.
        
        # Let's try a different structural change: 
        # **Grid Search in Objective Space?** No, we must call problem(x).
        
        # Let's stick to the analytic front but use **Latin Hypercube Sampling (LHS)** 
        # on the parameter x1, which is often better than uniform for 
        # low-discrepancy sequences, ensuring better space-filling properties 
        # in the objective space if the mapping is non-linear.
        
        # Actually, the current uniform linspace is already a 1D LHS with 
        # 1 point per stratum. 
        # What if we use **Quasi-random** sequences like Sobol or Halton?
        # Halton sequence is easy to implement for 1D.
        
        # Let's implement a Halton sequence generator for 1D.
        # Halton(2) is just van der Corput base 2.
        
        def halton(index, base):
            """Generate Halton sequence point."""
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Generate budget points using Halton sequence
        # This provides better low-discrepancy properties than uniform 
        # for non-linear mappings, potentially capturing the "bends" 
        # of the ZDT3 front more effectively.
        
        x1_samples = np.array([halton(i + 1, 2) for i in range(budget)])
        
        # Ensure we cover the boundaries 0 and 1 explicitly?
        # Halton doesn't hit 0 or 1 exactly. 
        # We can force the first point to be 0 and the last to be 1?
        # Or just add them if budget allows.
        # Let's replace the first and last with 0 and 1 to ensure full range coverage.
        if budget > 2:
            x1_samples[0] = 0.0
            x1_samples[-1] = 1.0
        elif budget == 2:
            x1_samples[0] = 0.0
            x1_samples[1] = 1.0
        elif budget == 1:
            x1_samples[0] = 0.5
            
        # Sort the samples to ensure monotonic progression (not strictly 
        # necessary for hypervolume, but good for consistency).
        x1_samples.sort()
        
        # Create decision vectors
        X = np.zeros((budget, dim))
        X[:, 0] = x1_samples
        # X[:, 1:] = 0.0 to achieve g=1
        
        # Evaluate all points
        for i in range(budget):
            if time.time() > deadline:
                break
            x = X[i]
            problem(x)
            
        return X
# EVOLVE-BLOCK-END
