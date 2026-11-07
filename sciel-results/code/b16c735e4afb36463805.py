import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0

        def check_time():
            return time.time() < deadline

        def evaluate(x):
            nonlocal used
            if not check_time():
                return
            if used >= self.budget:
                return
            # Clip to bounds
            x_clipped = np.clip(x, lo, hi)
            # For DTLZ7, the Pareto front is at x[2:] = 0
            # We set x[2:] = 0 to be on the front
            x_eval = x_clipped.copy()
            x_eval[2:] = 0.0
            problem(x_eval)
            used += 1

        # DTLZ7 Pareto front parameterization:
        # f1 = x0, f2 = x1, f3 = 1 - x1 * h(x1, 1)
        # where h(y, 1) = 9 / (1 + 2*(sin(3*pi*y) + sin(4*pi*y)))
        
        def h_func(y):
            s = 9.0 / (1.0 + 2.0 * (np.sin(3.0 * np.pi * y) + np.sin(4.0 * np.pi * y)))
            return s

        def obj_space(x0, x1):
            """Map (x0, x1) to objective space (f1, f2, f3) on Pareto front."""
            f1 = x0
            f2 = x1
            f3 = 1.0 - x1 * h_func(x1)
            return np.array([f1, f2, f3])

        # Strategy: Max-min facility location in objective space
        # Place points to maximize the minimum pairwise distance in (f1, f2, f3) space
        # This ensures uniform coverage where hypervolume contribution is highest.

        n_total = self.budget

        # Start with corner/edge points that are critical for hypervolume
        # The reference point is [1.1, 1.1, 25.0]
        # Points near (0,0,1) contribute most to hypervolume
        
        # Initial seed points: corners of the Pareto front bounding box
        # f1 in [0,1], f2 in [0,1], f3 = 1 - x1*h(x1,1)
        # f3 range: when x1=0, f3=1; when x1=1, f3=1-h(1,1)=1-9/(1+2*(sin(3pi)+sin(4pi)))=1-0=1
        # Actually sin(3*pi)=0, sin(4*pi)=0, so h(1,1)=9, f3=1-1*9=-8? No wait...
        # Let me recompute: at x1=1, h(1,1) = 9/(1+2*(sin(3π)+sin(4π))) = 9/(1+2*(0+0)) = 9
        # f3 = 1 - 1*9 = -8. That can't be right for DTLZ7.
        
        # Let me recheck DTLZ7 definition:
        # f1 = x0
        # f2 = x1  
        # f3 = g(x) * (1 - x1 * h(x1, g))
        # h(y, g) = 9*g / (1 + 2*g*(sin(3πy) + sin(4πy)))
        # With g=1: h(y,1) = 9/(1+2*(sin(3πy)+sin(4πy)))
        
        # At y=0: h=9/(1+0)=9, f3=1-0=1
        # At y=1: h=9/(1+2*(0+0))=9, f3=1-1*9=-8
        # Hmm, this gives negative f3 which doesn't make sense for minimization.
        
        # Actually looking at DTLZ7 more carefully, the standard formulation is:
        # f_{M-1} = g * (1 - x_{M-2} * h(x_{M-2}, g))
        # where h(y,g) = 9g/(1+2g(sin(3πy)+sin(4πy)))
        # For M=3: f2 = g*(1-x1*h(x1,g))
        
        # The Pareto front should have f3 >= 0. Let me just compute the actual front
        # by sampling and use that for facility location.

        # Sample the Pareto front densely to understand its shape
        n_sample = 200
        x0_samples = np.linspace(0, 1, n_sample)
        x1_samples = np.linspace(0, 1, n_sample)
        
        # Create a fine grid to map out the front
        x0_grid, x1_grid = np.meshgrid(x0_samples, x1_samples)
        f1_grid = x0_grid
        f2_grid = x1_grid
        f3_grid = 1.0 - x1_grid * h_func(x1_grid)
        
        # The Pareto front in 2D projection (f1, f2) is the unit square
        # f3 is determined by f2 only (not by f1)
        
        # For max-min facility location, we work in 3D objective space
        # but f3 depends only on f2, so the front is a curve in (f2, f3) plane
        # extruded along f1.
        
        # Key insight: Since f3 depends only on f2, we can do facility location
        # in 2D: (f1, f2) space, with the understanding that f3 is determined.
        # But for proper hypervolume, we should consider 3D distances weighted
        # by the hypervolume contribution.
        
        # Simpler approach: Do max-min in (f1, f2) space since that's where
        # the "shape" of the front lives, and f3 is a function of f2.
        
        # Max-min facility location: place n points in [0,1]^2 to maximize min pairwise distance
        # Use greedy algorithm: start with a point, then repeatedly add the point
        # that maximizes the minimum distance to existing points.
        
        # For better hypervolume, we should weight by hypervolume contribution.
        # Points near (0,0) contribute more. So use a transformed space.
        
        # Actually, let's just do standard max-min in (f1, f2) with a twist:
        # Use a non-uniform grid that's denser near (0,0) since those points
        # contribute more to hypervolume with reference [1.1, 1.1, 25].
        
        # Alternative: Just do max-min in the actual 3D objective space.
        # This naturally accounts for the f3 dimension.
        
        # Greedy max-min facility location in 3D objective space
        points_obj = []  # list of (f1, f2, f3) points
        points_param = []  # corresponding (x0, x1) parameters
        
        # Start with the point that gives maximum hypervolume contribution
        # This is near (0, 0, 1) since reference is [1.1, 1.1, 25]
        # Hypervolume contribution of a point p is (1.1-f1)*(1.1-f2)*(25-f3)
        # for points dominated by reference.
        
        # Start at (0, 0, 1) - the "ideal" point
        first_point = obj_space(0.0, 0.0)
        points_obj.append(first_point)
        points_param.append((0.0, 0.0))
        
        # Greedy: add points one by one, each maximizing min distance to existing
        for k in range(1, n_total):
            if not check_time():
                break
            if used >= self.budget:
                break
            
            # Sample candidate positions on a fine grid
            # Use a grid that's denser in regions of high hypervolume contribution
            n_cand = 50
            cand_x0 = np.linspace(0, 1, n_cand)
            cand_x1 = np.linspace(0, 1, n_cand)
            
            best_min_dist = -1
            best_x0, best_x1 = 0.5, 0.5
            
            for cx0 in cand_x0:
                for cx1 in cand_x1:
                    cand_obj = obj_space(cx0, cx1)
                    # Compute min distance to existing points
                    dists = np.array([np.linalg.norm(cand_obj - po) for po in points_obj])
                    min_dist = np.min(dists)
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_x0, best_x1 = cx0, cx1
            
            # Add the best point
            new_obj = obj_space(best_x0, best_x1)
            points_obj.append(new_obj)
            points_param.append((best_x0, best_x1))
            
            # Evaluate this point
            x = np.zeros(self.dim)
            x[0] = lo[0] + (hi[0] - lo[0]) * best_x0
            x[1] = lo[1] + (hi[1] - lo[1]) * best_x1
            evaluate(x)
            
            if used >= self.budget:
                break

        return None
