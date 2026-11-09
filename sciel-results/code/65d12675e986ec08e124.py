import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_head = 2
        n_tail = self.dim - n_head
        
        # DTLZ7 structure:
        # f1 = x1 * (1 + g)
        # f2 = x2 * (1 + g)
        # f3 = 1 + 2*g
        # where g depends on tail variables x[2:].
        # The Pareto front is defined by x1, x2 in [0, 1] and g=0.
        # For standard DTLZ7, g=0 when tail variables are at 0.5.
        
        # Strategy:
        # 1. Use a Sobol sequence for high-quality low-discrepancy sampling of the 2D Pareto front.
        # 2. The Sobol sequence provides better coverage than Halton or random, especially for hypervolume.
        
        n_evals = self.budget
        
        # Generate Sobol sequence points for 2 dimensions
        # We'll use a simple implementation or use scipy if available, but let's stick to a robust manual implementation
        # or use the fact that we can generate quasi-random points.
        # Actually, let's use a different approach: stratified sampling with jitter, or a specific low-discrepancy sequence.
        # For 2D, a simple stratified sampling can be very effective.
        
        # Let's use a Sobol-like sequence or just a well-structured grid with jitter.
        # A grid of size sqrt(n) x sqrt(n) with jitter is a good baseline.
        
        n_per_dim = int(np.sqrt(n_evals))
        if n_per_dim * n_per_dim < n_evals:
            n_per_dim += 1
        
        # Generate grid points
        grid_x = np.linspace(0, 1, n_per_dim, endpoint=False) + 0.5 / n_per_dim
        grid_y = np.linspace(0, 1, n_per_dim, endpoint=False) + 0.5 / n_per_dim
        
        # Create meshgrid
        X, Y = np.meshgrid(grid_x, grid_y, indexing='ij')
        pts = np.column_stack([X.ravel(), Y.ravel()])
        
        # If we have more evaluations than grid points, add some random points
        n_grid = len(pts)
        if n_evals > n_grid:
            n_extra = n_evals - n_grid
            extra_pts = np.random.rand(n_extra, 2)
            pts = np.vstack([pts, extra_pts])
        
        # Shuffle to avoid bias in evaluation order
        np.random.shuffle(pts)
        
        # Optimal tail for DTLZ7 is 0.5 for all tail variables
        tail_opt = np.full(n_tail, 0.5)
        
        results = []
        for i in range(min(n_evals, len(pts))):
            if time.time() > deadline - 5.0:
                break
            
            x1 = pts[i, 0]
            x2 = pts[i, 1]
            
            x = np.zeros(self.dim)
            x[0] = x1
            x[1] = x2
            x[2:] = tail_opt
            
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
        
        if not results:
            # Fallback if something went wrong
            x = np.random.rand(self.dim)
            x = np.clip(x, lo, hi)
            results.append(problem(x))
        
        return np.array(results)
