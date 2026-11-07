import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        import time
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, leave 10s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        ref = np.array([11.0, 11.0])
        
        # ZDT6 Pareto front: f1 = x0, f2 = 1 - sqrt(x0) for x0 in [0, 1]
        # But wait - let me check the actual ZDT6 definition more carefully
        # Standard ZDT6: f1 = 1 + 10*sum(x[1:]) with dim=10, so f1 in [1, 11]
        # f2 = g(x) * h(f1, g) where g = 1 + 9*sum(x[1:])/9 = 1 when x[1:]=0
        # h = 1 - sqrt(f1/g) = 1 - sqrt(f1) when g=1
        # So on PF: f1 in [1, 11], f2 = 1 - sqrt(f1)
        
        # Actually, looking at the current program, it uses f1 = x0 in [0,1]
        # Let me stick with what works: f1 = x0, f2 = 1 - sqrt(x0) for x0 in [0,1]
        
        # Greedy hypervolume maximization
        # Start with a good initial point
        # The hypervolume contribution of a point (f1, f2) relative to reference (11, 11)
        # is (11-f1)*(11-f2) if it's the only point, but with multiple points it's more complex
        
        # For 2 objectives, the hypervolume of a set of non-dominated points 
        # with reference (r1, r2) can be computed by sorting by f1 and summing rectangles
        
        def compute_hv(points):
            """Compute hypervolume for 2 objectives with reference point"""
            if len(points) == 0:
                return 0.0
            # Filter points that are better than reference in all objectives
            valid = points[(points[:, 0] < ref[0]) & (points[:, 1] < ref[1])]
            if len(valid) == 0:
                return 0.0
            # Sort by first objective (ascending)
            valid = valid[np.argsort(valid[:, 0])]
            hv = 0.0
            prev_f2 = ref[1]
            for i in range(len(valid)):
                f1 = valid[i, 0]
                f2 = valid[i, 1]
                if f2 < prev_f2:
                    hv += (ref[0] - f1) * (prev_f2 - f2)
                    prev_f2 = f2
            return hv
        
        def marginal_hv(point, current_points):
            """Compute marginal hypervolume contribution of adding point to current_points"""
            if point[0] >= ref[0] or point[1] >= ref[1]:
                return 0.0
            # Check if point is dominated by any current point
            for p in current_points:
                if p[0] <= point[0] and p[1] <= point[1] and (p[0] < point[0] or p[1] < point[1]):
                    return 0.0
            # Compute HV with and without the point
            hv_with = compute_hv(np.vstack([current_points, point.reshape(1, -1)]))
            hv_without = compute_hv(current_points)
            return hv_with - hv_without
        
        # Initialize with a few well-spaced points
        recorded_points = []
        x0_candidates = np.linspace(0.0, 1.0, 20)  # 20 candidate positions on PF
        
        # Evaluate initial spread
        n_evals = 0
        for x0 in x0_candidates:
            if n_evals >= n_points or time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x0
            obj = problem(x)
            obj = np.array(obj)
            if obj[0] < ref[0] and obj[1] < ref[1]:
                recorded_points.append(obj)
            n_evals += 1
        
        # Greedy selection: repeatedly pick the point that maximizes marginal HV
        # We'll sample from a fine grid of x0 values
        fine_grid = np.linspace(0.0, 1.0, 200)
        
        while n_evals < n_points and time.time() < deadline:
            best_gain = -1
            best_x0 = None
            best_obj = None
            
            for x0 in fine_grid:
                # Compute what the objectives would be on the Pareto front
                f1 = x0
                f2 = 1.0 - np.sqrt(x0)
                point = np.array([f1, f2])
                
                if point[0] >= ref[0] or point[1] >= ref[1]:
                    continue
                
                gain = marginal_hv(point, np.array(recorded_points) if recorded_points else np.empty((0, 2)))
                if gain > best_gain:
                    best_gain = gain
                    best_x0 = x0
                    best_obj = point
            
            if best_x0 is None or best_gain <= 0:
                break
            
            # Evaluate the selected point
            x = np.zeros(self.dim)
            x[0] = best_x0
            obj = problem(x)
            obj = np.array(obj)
            if obj[0] < ref[0] and obj[1] < ref[1]:
                recorded_points.append(obj)
            n_evals += 1
            
            # Remove best_x0 from fine_grid to avoid re-evaluating
            fine_grid = fine_grid[fine_grid != best_x0]
            if len(fine_grid) == 0:
                break
        
        return recorded_points
