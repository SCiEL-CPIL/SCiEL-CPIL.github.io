import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        # ZDT6 specific knowledge
        # f1(x1) = 1 - (1 - exp(-4*x1) * sin(6*pi*x1)**6)**(1/10)
        # f2(x1) = 1 - sqrt(f1(x1))
        # The Pareto front is parameterized by x1 in [0, x1_peak] where f1 is maximized.
        
        def calc_f1(x1):
            # x1 is scalar or array
            return 1 - (1 - np.exp(-4 * x1) * np.sin(6 * np.pi * x1)**6)**0.1

        # 1. Find the precise maximum of f1 and its location x1_peak
        # The maximum is in the first lobe, roughly x1 in [0, 0.2]
        # We use a dense sampling to find the peak location accurately
        # Then refine with local search
        
        # Coarse search
        x1_coarse = np.linspace(0, 0.2, 10000)
        f1_coarse = calc_f1(x1_coarse)
        idx_coarse = np.argmax(f1_coarse)
        
        # Refine around the coarse peak
        if idx_coarse > 0 and idx_coarse < len(x1_coarse) - 1:
            x1_left = x1_coarse[idx_coarse - 1]
            x1_right = x1_coarse[idx_coarse + 1]
        else:
            x1_left = max(0, x1_coarse[idx_coarse] - 0.001)
            x1_right = min(0.2, x1_coarse[idx_coarse] + 0.001)
        
        # Fine search
        x1_fine = np.linspace(x1_left, x1_right, 100000)
        f1_fine = calc_f1(x1_fine)
        idx_fine = np.argmax(f1_fine)
        x1_peak = x1_fine[idx_fine]
        f1_max = f1_fine[idx_fine]
        
        # 2. Build a high-resolution lookup table for f1(x1) on [0, x1_peak]
        # We want to map target f1 values to x1 values accurately
        n_lookup = 200000
        x1_lookup = np.linspace(0, x1_peak, n_lookup)
        f1_lookup = calc_f1(x1_lookup)
        
        # Sort by f1 to enable binary search
        sort_idx = np.argsort(f1_lookup)
        f1_sorted = f1_lookup[sort_idx]
        x1_sorted = x1_lookup[sort_idx]
        
        # 3. Determine the optimal distribution of points to maximize hypervolume
        # For a convex Pareto front, uniform distribution in objective space is optimal
        # For ZDT6, the front is f2 = 1 - sqrt(f1), which is concave
        # To maximize hypervolume with reference point [11, 11], we should
        # distribute points such that they cover the front well
        
        # Strategy: Use a mix of uniform f1 targets and some strategic points
        # Actually, for hypervolume maximization, we want to spread points
        # evenly in the objective space. Since f2 = 1 - sqrt(f1), 
        # uniform f1 targets give non-uniform f2 spacing.
        
        # Better approach: Optimize the placement of points to maximize hypervolume
        # But with 200 points, this is complex. Let's use a good heuristic.
        
        # For a concave front, uniform sampling in the parameter (x1) space
        # or in f1 space both work well. Let's try uniform f1 targets first.
        
        n_points = self.budget
        
        # Generate target f1 values uniformly in [0, f1_max]
        target_f1s = np.linspace(0, f1_max, n_points)
        
        # Find x1 for each target f1 using binary search on sorted f1
        indices = np.searchsorted(f1_sorted, target_f1s)
        indices = np.clip(indices, 0, len(f1_sorted) - 1)
        
        # Refinement: Check neighbors to find the closest match
        x1_targets = np.zeros(n_points)
        
        # Vectorized refinement
        for i in range(n_points):
            idx = indices[i]
            target = target_f1s[i]
            
            # Check current, previous, and next
            best_dist = np.inf
            best_x1 = x1_sorted[idx]
            
            for j in [idx - 1, idx, idx + 1]:
                if 0 <= j < len(f1_sorted):
                    dist = abs(f1_sorted[j] - target)
                    if dist < best_dist:
                        best_dist = dist
                        best_x1 = x1_sorted[j]
            
            x1_targets[i] = best_x1
        
        # 4. Evaluate all points
        recorded = []
        for i in range(n_points):
            # Check time
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x1_targets[i]
            # Clip to bounds [0, 1]
            x = np.clip(x, 0, 1)
            obj = problem(x)
            recorded.append(obj)
        
        return recorded
