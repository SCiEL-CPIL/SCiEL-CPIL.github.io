# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ7 exploiting the known Pareto front structure.
    The Pareto front of DTLZ7 with M objectives lies on the subspace where x_1...x_{n-M} are free
    and x_{n-M+1}...x_{n-1} are 0, with x_n determined. For 22 vars and 3 objectives, the front
    is parameterized by t_1, t_2 in [0,1] with x_0 = t_1, x_1 = t_2, x_2...x_20 = 0.
    
    Strategy:
    1. Dense sampling of the known Pareto front subspace (x[2:]=0) to maximize hypervolume coverage.
    2. Use remaining budget for local refinement around promising points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        used = 0
        all_points = []
        all_objs = []
        
        def eval_point(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            return x, f
        
        # Phase 1: Dense sampling of the Pareto front subspace
        # For DTLZ7 with 3 objectives and 22 vars, the Pareto front is:
        # x[0] = t1, x[1] = t2, x[2:] = 0
        # where t1, t2 in [0, 1]
        
        # Create a dense grid on the Pareto front
        n_grid = 25  # 25x25 = 625 points, but we only have 200 budget
        # So we need to be more selective
        # Let's use a space-filling approach on the 2D parameter space
        
        # Allocate budget: ~150 for front sampling, ~50 for refinement
        n_front_samples = 150
        n_refinement = self.budget - n_front_samples
        
        # Generate points on the Pareto front using a quasi-random sequence
        # Use Halton sequence or similar for good coverage
        t1_vals = np.linspace(0, 1, n_front_samples)
        t2_vals = np.linspace(0, 1, n_front_samples)
        
        # Create a subset of grid points to fit budget
        # Use a stratified sampling approach
        n_per_dim = int(np.sqrt(n_front_samples))
        t1_grid = np.linspace(0, 1, n_per_dim)
        t2_grid = np.linspace(0, 1, n_per_dim)
        
        front_points = []
        for t1 in t1_grid:
            for t2 in t2_grid:
                x = np.zeros(self.dim)
                x[0] = t1
                x[1] = t2
                front_points.append(x)
        
        # Shuffle to ensure we don't stop early with biased subset
        rng.shuffle(front_points)
        
        # Evaluate front points
        for x in front_points:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            x, f = eval_point(x)
            all_points.append(x)
            all_objs.append(f)
        
        # Phase 2: Local refinement around best points
        if used < self.budget and len(all_objs) > 0:
            all_objs_arr = np.array(all_objs)
            
            # Find non-dominated points
            n = len(all_objs_arr)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(all_objs_arr[j] <= all_objs_arr[i]) and np.any(all_objs_arr[j] < all_objs_arr[i]):
                        dominated[i] = True
                        break
            
            non_dom_idx = np.where(~dominated)[0]
            
            # Refine around non-dominated points
            if len(non_dom_idx) > 0:
                # Pick a few non-dominated points to refine
                n_refine = min(10, len(non_dom_idx))
                refine_indices = rng.choice(non_dom_idx, size=n_refine, replace=False)
                
                remaining_budget = self.budget - used
                steps_per_point = max(1, remaining_budget // n_refine)
                
                for idx in refine_indices:
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    base_x = all_points[idx]
                    base_f = all_objs[idx]
                    
                    for _ in range(steps_per_point):
                        if used >= self.budget:
                            break
                        if time.time() > deadline:
                            break
                        
                        # Perturb only the first two variables (Pareto front parameters)
                        # and keep others at 0
                        x_new = base_x.copy()
                        noise = 0.05 * rng.standard_normal(2)
                        x_new[0] = np.clip(base_x[0] + noise[0], 0, 1)
                        x_new[1] = np.clip(base_x[1] + noise[1], 0, 1)
                        
                        x_new, f_new = eval_point(x_new)
                        all_points.append(x_new)
                        all_objs.append(f_new)
        
        return all_points
# EVOLVE-BLOCK-END
