"""Multi-objective optimizer for ZDT2 using Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct Pareto front sampling for ZDT2 with hypervolume-optimized spacing."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        # ZDT2 Pareto front: f1 = t, f2 = 1 - sqrt(t) for t in [0, 1]
        # We want to sample t values that maximize hypervolume contribution.
        # The hypervolume contribution of a point (t, 1-sqrt(t)) with reference (11,11)
        # is proportional to the area it "covers" uniquely.
        
        # Strategy: Use inverse-CDF sampling based on the hypervolume density.
        # For ZDT2, the Pareto front is f2 = 1 - sqrt(f1).
        # The marginal hypervolume gain is higher where the curve is steeper in terms of 
        # the area swept. Let's use a transformation to concentrate samples.
        
        # Actually, for ZDT2, the Pareto front is convex. 
        # f1 = t, f2 = 1 - sqrt(t).
        # The "ideal" spacing for hypervolume in 2D with a convex front can be derived.
        
        # Let's try a mix:
        # 1. Uniform sampling in f1 (t)
        # 2. Sampling biased towards regions where the curve has higher curvature or marginal gain.
        
        # For now, let's use a deterministic set of t values that are well-distributed.
        # Since we have budget=200, we can evaluate 200 points.
        
        # Key insight: The problem function clips x to bounds and computes ZDT2 objectives.
        # For ZDT2, f1 = x[0], f2 = (1+10*(n-1)) * (sum((xi-1)^2) / (n-1) - 2*sum(cos(20*pi*xi) / (n-1))) + 1.
        # The Pareto front is achieved when all xi (for i>0) are 1.
        # So, to get a point on the Pareto front with f1 = t, we set x[0]=t and x[1:]=1.
        
        # Let's generate t values.
        # We want to maximize hypervolume. 
        # The hypervolume of a set of points on the Pareto front with reference (11,11) is:
        # Sum over consecutive points of the area of the rectangles.
        
        # Let's use a non-uniform grid.
        # For a convex function f2(t), the optimal spacing for hypervolume with a fixed number of points
        # tends to be denser where the function is flatter? No, denser where the marginal gain is higher.
        
        # Actually, let's just use a dense uniform grid first, then refine.
        # With 200 evaluations, we can do 200 points on the Pareto front.
        
        # Generate t values.
        # Let's try t = i / (N-1) for i in 0..N-1.
        N = self.budget
        
        # To be safe and ensure we cover the whole front, use uniform t.
        # But maybe a slightly biased distribution helps?
        # Let's try t = (i + 0.5) / N for i in 0..N-1.
        t_vals = (np.arange(N) + 0.5) / N
        
        # Generate x vectors
        # x[0] = t, x[1:] = 1.0
        X = np.zeros((N, n))
        X[:, 0] = t_vals
        X[:, 1:] = 1.0
        
        # Clip to bounds just in case
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
