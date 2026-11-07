import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 300.0 # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initial Population: Halton Sequence
        n_init = 50
        X = self._halton(n_init, self.dim)
        X = lo + X * (hi - lo)
        
        archive_X = []
        archive_F = []
        evaluations = 0
        
        def add_point(x, f):
            nonlocal evaluations
            evaluations += 1
            # Check if dominated by any point in archive
            for f_arch in archive_F:
                if np.all(f_arch <= f) and np.any(f_arch < f):
                    return
            # Remove points dominated by new point
            new_archive_X = []
            new_archive_F = []
            for x_arch, f_arch in zip(archive_X, archive_F):
                if not (np.all(f <= f_arch) and np.any(f < f_arch)):
                    new_archive_X.append(x_arch)
                    new_archive_F.append(f_arch)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X[:] = new_archive_X
            archive_F[:] = new_archive_F

        # Evaluate initial population
        for x in X:
            if time.time() - start_time > deadline:
                break
            f = problem(x)
            add_point(x, f)
            
        # Main Loop
        while evaluations < self.budget:
            if time.time() - start_time > deadline:
                break
            
            if not archive_F:
                break
            
            # Select parent based on hypervolume contribution
            parent_idx = self._select_parent_hv(archive_F, [3000.0, 0.06])
            parent_x = archive_X[parent_idx]
            
            # Local Search Strategy
            # We try to improve the point using Nelder-Mead on a scalarized objective
            # The weight is based on the local slope of the Pareto front
            # For 2 objectives, we can just try minimizing a weighted sum
            # Let's pick a weight that balances the objectives based on their current values
            f_parent = archive_F[parent_idx]
            # Normalize weights by current values to make them comparable
            w1 = 1.0 / (f_parent[0] + 1e-9)
            w2 = 1.0 / (f_parent[1] + 1e-9)
            w = np.array([w1, w2])
            w = w / np.sum(w)
            
            def objective(x_local):
                f_local = problem(x_local)
                return np.dot(w, f_local)
                
            # Nelder-Mead
            try:
                res = minimize(objective, parent_x, method='Nelder-Mead', 
                               options={'maxiter': 10, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
                x_new = np.clip(res.x, lo, hi)
                f_new = problem(x_new)
                add_point(x_new, f_new)
            except:
                # Fallback to mutation
                x_new = parent_x + 0.1 * (hi - lo) * np.random.normal(size=self.dim)
                x_new = np.clip(x_new, lo, hi)
                f_new = problem(x_new)
                add_point(x_new, f_new)
                
        return archive_X

    def _halton(self, n, dim):
        # Simple Halton sequence generator
        points = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2
            for i in range(n):
                f = 1.0
                i_val = i + 1
                while i_val > 0:
                    f /= base
                    i_val, rem = divmod(i_val, base)
                    points[i, d] += rem * f
        return points

    def _select_parent_hv(self, archive_F, ref_point):
        # Calculate hypervolume contribution for each point
        # For 2 objectives, sort by f1
        n = len(archive_F)
        if n == 0:
            return 0
        if n == 1:
            return 0
            
        # Sort indices by f1
        idx = np.argsort(archive_F[:, 0])
        sorted_F = archive_F[idx]
        
        # Calculate contribution of each point
        # Contribution of point i is the area of the rectangle it uniquely covers
        # This is complex to calculate exactly for each point in O(N log N)
        # Approximation: The point that is "most central" or has the largest gap to neighbors
        # Simpler: Just pick a random non-dominated point from the top half of hypervolume
        # Or, calculate the "shapley value" approximation
        
        # For simplicity and speed, let's just pick the point with the best "normalized" objective sum
        # Or, pick a point that is not an extreme (not min f1 or min f2) to explore the middle
        f1_min = np.min(archive_F[:, 0])
        f1_max = np.max(archive_F[:, 0])
        f2_min = np.min(archive_F[:, 1])
        f2_max = np.max(archive_F[:, 1])
        
        # Normalize
        norm_F = np.zeros_like(archive_F)
        norm_F[:, 0] = (archive_F[:, 0] - f1_min) / (f1_max - f1_min + 1e-9)
        norm_F[:, 1] = (archive_F[:, 1] - f2_min) / (f2_max - f2_min + 1e-9)
        
        # Distance to (1,1) in normalized space? No, we want to minimize objectives.
        # So distance to (0,0) in normalized space?
        # Let's pick the point closest to the center of the normalized box (0.5, 0.5)
        # This helps explore the "knee"
        dist = np.sqrt((norm_F[:, 0] - 0.5)**2 + (norm_F[:, 1] - 0.5)**2)
        return idx[np.argmin(dist)]
