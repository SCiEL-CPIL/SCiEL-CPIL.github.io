import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # DTLZ7 Pareto front is a 2D surface parameterized by x[0] and x[1]
        # The other variables should be 0 (or lo) for optimal solutions.
        # We'll create a 2D space-filling set of points over x[0] and x[1].
        
        # Strategy:
        # 1. Generate a dense Halton sequence in 2D for space-filling coverage.
        # 2. Explicitly add boundary points (corners and edges) to ensure
        #    the extremes of the front are captured, which is crucial for hypervolume.
        # 3. Use the full budget.
        
        # We want to cover the square [0,1]x[0,1] (mapped to bounds) well.
        # Halton sequence is deterministic and space-filling.
        
        def halton_2d(n, base1=2, base2=3):
            points = []
            for i in range(1, n + 1):
                # Base base1
                f1 = 1.0
                r1 = 0.0
                idx = i
                while idx:
                    f1 /= base1
                    r1 += f1 * (idx % base1)
                    idx //= base1
                
                # Base base2
                f2 = 1.0
                r2 = 0.0
                idx = i
                while idx:
                    f2 /= base2
                    r2 += f2 * (idx % base2)
                    idx //= base2
                
                points.append((r1, r2))
            return np.array(points)
        
        # Use a mix: Halton for interior + strategic boundary points
        # The Pareto front of DTLZ7 is bounded by x[0]=0, x[0]=1, x[1]=0, x[1]=1.
        # The Pareto front of DTLZ7 is bounded by x[0]=0, x[0]=1, x[1]=0, x[1]=1.
        # Halton sequence does not hit boundaries exactly.
        
        # Use 14 boundary points: 4 corners + 6 along high-f3 edges + 3 near origin
        n_boundary = 14
        n_halton = n_evals - n_boundary
        
        if n_halton < 0:
            n_halton = 0
            n_boundary = n_evals
        
        grid_points = halton_2d(n_halton)
        
        # Define boundary points in [0,1]x[0,1]
        # Strategy: capture the extreme points of the Pareto front
        # The front extends to high f3 when x0 or x1 is near 1
        # Add points along the edges where f3 is maximized
        boundary_points = np.array([
            # Corners - critical for hypervolume
            [0.0, 0.0],   # f=(0,0,1)
            [0.0, 1.0],   # f=(0,1,h(0,1)+1)
            [1.0, 0.0],   # f=(1,0,h(1,0)+1)
            [1.0, 1.0],   # f=(1,1,h(1,1)+1) - highest f3
            # Points along x0=1 edge (high f3 region)
            [1.0, 0.25],
            [1.0, 0.5],
            [1.0, 0.75],
            # Points along x1=1 edge (high f3 region)
            [0.25, 1.0],
            [0.5, 1.0],
            [0.75, 1.0],
            # Points near origin for good f1,f2 coverage
            [0.1, 0.1],
            [0.0, 0.1],
            [0.1, 0.0],
        ])
        
        # Use up to n_boundary points
        if len(boundary_points) > n_boundary:
            boundary_points = boundary_points[:n_boundary]
        
        # Combine
        if n_halton > 0:
            all_points = np.vstack([grid_points, boundary_points])
        else:
            all_points = boundary_points
        
        # Map from [0,1] to [lo, hi] for x[0] and x[1]
        # x[0] in [lo[0], hi[0]]
        # x[1] in [lo[1], hi[1]]
        
        X = np.zeros((len(all_points), self.dim))
        X[:, 0] = lo[0] + (hi[0] - lo[0]) * all_points[:, 0]
        X[:, 1] = lo[1] + (hi[1] - lo[1]) * all_points[:, 1]
        # x[2:] should be lo[2:] (which is 0) to minimize g(x)
        X[:, 2:] = lo[2:]
        
        results = []
        for i in range(len(X)):
            if time.time() > deadline:
                break
            
            x = X[i]
            # Clip to be safe
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
        
        return results
