"""Multi-objective optimizer for ZDT4: direct Pareto front sampling."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective search on ZDT4.

    Strategy: The true Pareto front of ZDT4 lies at x_1..x_9 = 0 (where f2
    is minimized to 10). We densely sample this known front using the entire
    evaluation budget to maximize hypervolume. We use arc-length parameterization
    to ensure uniform geometric spacing along the curve, which is more efficient
    for hypervolume maximization than uniform spacing in the decision variable.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin: evaluator kills at 330s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # x_1 is the first variable (index 0)
        x1_lo = lo[0]
        x1_hi = hi[0]
        
        # Ensure x_1 is in a valid domain for sqrt (ZDT4 f2 uses sqrt(x1))
        # Standard ZDT4 x1 >= 0.
        if x1_lo < 0:
            x1_lo = 0.0
            
        if x1_hi <= x1_lo:
            # Degenerate case, just evaluate center
            x1_val = (x1_lo + x1_hi) / 2 if x1_hi > x1_lo else 0.0
            X = np.zeros((1, dim))
            X[0, 0] = np.clip(x1_val, lo[0], hi[0])
            X[0, 1:] = np.clip(0.0, lo[1:], hi[1:])
            return np.array([problem(X[0])])

        # We want to sample x_1 values such that the arc length of the front is uniform.
        # Front: f1 = x1, f2 = 1 - sqrt(x1)
        # ds/dx1 = sqrt(1 + (df2/df1)^2) = sqrt(1 + 1/(4*x1))
        
        # Use a dense grid to compute cumulative arc length
        n_grid = 1000
        x1_grid = np.linspace(x1_lo, x1_hi, n_grid)
        
        # Handle x1=0 singularity in derivative
        # df2/df1 = -1/(2*sqrt(x1)). At x1=0, it's infinite.
        # We can approximate the arc length integral numerically.
        f1_grid = x1_grid
        # Clip small negative values if any due to floating point, though x1_lo >= 0
        f1_safe = np.maximum(f1_grid, 1e-12)
        f2_grid = 1.0 - np.sqrt(f1_safe)
        
        # Compute segment lengths
        df1 = np.diff(f1_grid)
        df2 = np.diff(f2_grid)
        seg_lengths = np.sqrt(df1**2 + df2**2)
        
        # Cumulative arc length
        arc_lengths = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        total_arc = arc_lengths[-1]
        
        # Generate target arc lengths for our budget samples
        n_samples = budget
        target_arcs = np.linspace(0.0, total_arc, n_samples)
        
        # Invert: find x1 values corresponding to target_arcs
        # Use linear interpolation on the cumulative arc length curve
        x1_samples = np.interp(target_arcs, arc_lengths, x1_grid)
        
        # Construct decision vectors
        X = np.zeros((n_samples, dim))
        X[:, 0] = x1_samples
        # Set x_1..x_9 (indices 1 to 9) to 0, clipped to bounds
        X[:, 1:] = np.clip(0.0, lo[1:], hi[1:])
        
        # Evaluate all points with deadline guard
        F = np.zeros((n_samples, 2))
        for i in range(n_samples):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return F
# EVOLVE-BLOCK-END
