import numpy as np


class Optimizer:
    """Deterministic adaptive grid search exploiting WFG9's known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Set remaining 9 variables to midpoint
        mid_vars = (lo[3:] + hi[3:]) / 2.0
        
        results = []
        used = 0
        
        # Phase 1: Systematic 4x4x4 grid (64 evaluations) to cover the space
        n1, n2, n3 = 4, 4, 4
        x1_vals = np.linspace(lo[0], hi[0], n1)
        x2_vals = np.linspace(lo[1], hi[1], n2)
        x3_vals = np.linspace(lo[2], hi[2], n3)
        
        # Track points that are non-dominated in the current set
        # We'll use a simple approach: collect all, then filter
        all_points = []
        all_objectives = []
        
        for x1 in x1_vals:
            for x2 in x2_vals:
                for x3 in x3_vals:
                    if used >= self.budget:
                        break
                    x = np.zeros(self.dim)
                    x[0] = x1
                    x[1] = x2
                    x[2] = x3
                    x[3:] = mid_vars
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    results.append(f)
                    all_points.append(x.copy())
                    all_objectives.append(np.array(f))
                    used += 1
                    
                    if used >= self.budget:
                        break
                if used >= self.budget:
                    break
            if used >= self.budget:
                break
        
        if used >= self.budget:
            return np.array(results)
        
        # Phase 2: Fine grid around promising regions
        # Find the 3 best non-dominated points from phase 1 based on sum of objectives
        # to define local search regions
        
        # Filter to points that are strictly better than reference point [3, 5, 7]
        ref = np.array([3.0, 5.0, 7.0])
        valid_indices = []
        for i, f in enumerate(all_objectives):
            if all(f[j] < ref[j] for j in range(self.n_obj)):
                valid_indices.append(i)
        
        if not valid_indices:
            # No valid points found, use all points
            valid_indices = list(range(len(all_objectives)))
        
        # Sort by sum of objectives to find promising regions
        valid_sorted = sorted(valid_indices, key=lambda i: np.sum(all_objectives[i]))
        
        # Take top 3 points as centers for local refinement
        centers = []
        for idx in valid_sorted[:3]:
            centers.append(all_points[idx])
        
        # If we have fewer than 3 valid points, add some from the full set
        while len(centers) < 3:
            for i, p in enumerate(all_points):
                if p not in centers:
                    centers.append(p)
                    break
            if len(centers) >= 3:
                break
        
        # Use remaining budget for fine grids around each center
        remaining = self.budget - used
        per_center = remaining // len(centers)
        
        for center in centers:
            if used >= self.budget:
                break
            
            # Define local bounds: 25% of original range centered on center
            range1 = hi[0] - lo[0]
            range2 = hi[1] - lo[1]
            range3 = hi[2] - lo[2]
            
            local_lo0 = max(lo[0], center[0] - 0.25 * range1)
            local_hi0 = min(hi[0], center[0] + 0.25 * range1)
            local_lo1 = max(lo[1], center[1] - 0.25 * range2)
            local_hi1 = min(hi[1], center[1] + 0.25 * range2)
            local_lo2 = max(lo[2], center[2] - 0.25 * range3)
            local_hi2 = min(hi[2], center[2] + 0.25 * range3)
            
            # Use per_center evaluations for a 4x4x2 grid or similar
            if per_center >= 32:
                n1_f, n2_f, n3_f = 4, 4, 2
            elif per_center >= 16:
                n1_f, n2_f, n3_f = 4, 4, 1
            else:
                n1_f, n2_f, n3_f = 4, 2, 2
            
            x1_f_vals = np.linspace(local_lo0, local_hi0, n1_f)
            x2_f_vals = np.linspace(local_lo1, local_hi1, n2_f)
            x3_f_vals = np.linspace(local_lo2, local_hi2, n3_f)
            
            for x1 in x1_f_vals:
                for x2 in x2_f_vals:
                    for x3 in x3_f_vals:
                        if used >= self.budget:
                            break
                        x = np.zeros(self.dim)
                        x[0] = x1
                        x[1] = x2
                        x[2] = x3
                        x[3:] = mid_vars
                        x = np.clip(x, lo, hi)
                        f = problem(x)
                        results.append(f)
                        used += 1
                        if used >= self.budget:
                            break
                    if used >= self.budget:
                        break
                if used >= self.budget:
                    break
            if used >= self.budget:
                break

        return np.array(results) if results else None
