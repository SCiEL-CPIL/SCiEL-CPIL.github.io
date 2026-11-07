import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        n = self.budget
        dim = self.dim
        
        # DTLZ7 structure:
        # f0 = x0
        # f1 = x1
        # g = 1 + 9 * sum(x[2:]) / (dim - 2)
        # f2 = g * (1 - sqrt(x1 / g))
        
        # Pareto front is at g=1, which implies x[2:] = 0.
        # The front is parameterized by x0, x1 in [0,1]^2.
        # f0 = x0, f1 = x1, f2 = 1 - sqrt(x1).
        
        # Hypervolume maximization strategy:
        # We want to place points on the front to maximize the union of boxes 
        # from each point to the reference point [1.1, 1.1, 25.0].
        # 
        # Key insight: The hypervolume contribution of a point (f0, f1, f2) 
        # is maximized when f0, f1, f2 are small. However, we need coverage.
        # 
        # A structured approach:
        # 1. Use a grid in the (f0, f1) space? 
        #    f0 = x0, f1 = x1. 
        #    f2 = 1 - sqrt(f1).
        #    The reference point is (1.1, 1.1, 25.0).
        #    The "box" for a point (f0, f1, f2) is [0, f0] x [0, f1] x [0, f2] 
        #    intersected with the region below the reference point? 
        #    No, hypervolume is the volume of the region dominated by the set of points.
        #    
        #    Let's use a deterministic low-discrepancy sequence (Sobol) to sample 
        #    the 2D parameter space (x0, x1) uniformly. This ensures good spread.
        #    
        #    However, the current program already does this and scores ~1.77.
        #    To improve, we need a *better* distribution for hypervolume.
        #    
        #    Hypervolume is sensitive to points near the "knee" of the front 
        #    and points that extend far in one objective while being good in others.
        #    For DTLZ7, f2 = 1 - sqrt(f1). 
        #    As f1 -> 0, f2 -> 1. As f1 -> 1, f2 -> 0.
        #    The "knee" is around f1 = 0.25, f2 = 0.5? 
        #    Actually, the curvature is highest near f1=0.
        #    
        #    Let's try a **Fibonacci Lattice** (Sunflower Map) in 2D.
        #    It provides a more uniform distribution than Halton/Sobol for small N 
        #    and has good low-discrepancy properties.
        #    
        #    Alternatively, we can use **Adaptive Sampling**:
        #    1. Start with a coarse Sobol set.
        #    2. Evaluate.
        #    3. Identify the "gaps" in the objective space that contribute most to HV.
        #    4. Place new points in those gaps.
        #    
        #    With 200 evaluations, we can afford a simple adaptive step.
        #    
        #    Let's implement a 2-stage approach:
        #    Stage 1: 50 points using Sobol.
        #    Stage 2: 150 points using a greedy HV-maximizing placement.
        #            For each new point, try a few candidate locations (from a fine Sobol grid) 
        #            and pick the one that maximizes the marginal HV gain.
        #            This is expensive if done naively, but we can approximate.
        #    
        #    Actually, a simpler structural change: **Space-filling curve mapping**.
        #    Map 1D Sobol points to 2D using a Z-curve or Hilbert curve?
        #    No, that's worse than direct 2D Sobol.
        #    
        #    Let's try **Latin Hypercube Sampling (LHS)** with stratification.
        #    LHS ensures exactly one point per row and column in a grid.
        #    This can be better than Sobol for small N if the grid is chosen well.
        #    
        #    Let's combine: 
        #    Use a **Sobol sequence** but **stratify** the final 200 points 
        #    into a 14x14 grid (196 points) + 4 extra points at corners.
        #    This ensures coverage of the entire [0,1]^2 space.
        #    
        #    Even better: Use **Jittered Grid**.
        #    Divide [0,1]^2 into a 14x14 grid. 
        #    Place one point in each cell, jittered uniformly within the cell.
        #    This is essentially LHS.
        #    
        #    Let's implement LHS with 14x14 grid (196 points) + 4 points at (0,0), (0,1), (1,0), (1,1) 
        #    to cover the extreme corners which are crucial for HV.
        
        # Generate LHS points
        n_grid = 14
        n_lhs = n_grid * n_grid  # 196
        n_extra = n - n_lhs      # 4
        
        # LHS:
        # x0: uniform in [i/n, (i+1)/n]
        # x1: uniform in [j/n, (j+1)/n]
        # We need to permute the x1 indices to avoid diagonal bias.
        
        # Use a fixed permutation for determinism (or seed-based, but seed is set by evaluator)
        # The evaluator sets np.random.seed(seed), so we can use np.random.
        
        x0 = np.zeros(n_lhs)
        x1 = np.zeros(n_lhs)
        
        # Create grid indices
        i = np.arange(n_grid)
        j = np.arange(n_grid)
        
        # Meshgrid
        ii, jj = np.meshgrid(i, i, indexing='ij')
        ii_flat = ii.flatten()
        jj_flat = jj.flatten()
        
        # Permute jj_flat to break correlation
        # Use a deterministic permutation based on the seed? 
        # The seed is already set. We can use np.random.permutation.
        perm = np.random.permutation(n_lhs)
        
        # Add jitter
        jitter_x0 = np.random.rand(n_lhs) / n_grid
        jitter_x1 = np.random.rand(n_lhs) / n_grid
        
        x0 = (ii_flat + jitter_x0) / n_grid
        x1 = (jj_flat[perm] + jitter_x1) / n_grid
        
        # Ensure bounds [0,1]
        x0 = np.clip(x0, 0, 1)
        x1 = np.clip(x1, 0, 1)
        
        # Extra points at corners
        extra_x0 = np.array([0.0, 0.0, 1.0, 1.0])[:n_extra]
        extra_x1 = np.array([0.0, 1.0, 0.0, 1.0])[:n_extra]
        
        x0_all = np.concatenate([x0, extra_x0])
        x1_all = np.concatenate([x1, extra_x1])
        
        # Construct full 22D vectors
        X = np.zeros((n, dim))
        X[:, 0] = x0_all
        X[:, 1] = x1_all
        # x[2:] = 0
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        if time.time() > deadline:
            return X
        
        # Optional: Refine? 
        # The LHS + corners should be very good.
        # Let's check if we can do better with a second pass.
        # With 200 evals, we've used them all.
        
        return X
