import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        used = 0
        # Store all evaluated points? No, just the non-dominated ones for guidance.
        # But the evaluator records ALL. So we want to maximize the count of non-dominated points.
        
        nd_points = [] # List of x
        nd_f = []      # List of f
        
        def is_dominated(f_new, f_list):
            for f_i in f_list:
                if np.all(f_i <= f_new) and np.any(f_i < f_new):
                    return True
            return False

        def update_set(x, f):
            nonlocal nd_points, nd_f
            if is_dominated(f, nd_f):
                return False
            # Remove dominated points
            new_pts = []
            new_fs = []
            for xi, fi in zip(nd_points, nd_f):
                if not (np.all(f <= fi) and np.any(f < fi)):
                    new_pts.append(xi)
                    new_fs.append(fi)
            nd_points = new_pts
            nd_f = new_fs
            nd_points.append(x)
            nd_f.append(f)
            return True

        # 1. Initialization: Stratified Sampling
        # Try to cover the space
        n_init = 10
        for _ in range(n_init):
            if used >= self.budget or time.time() > deadline:
                break
            x = lo + span * np.random.rand(self.dim)
            f = problem(x)
            used += 1
            update_set(x, f)
            
        # 2. Local Search / Hill Climbing on non-dominated points
        # We will iterate through the non-dominated points and try to improve them.
        # To handle disconnected fronts, we also do random jumps.
        
        max_local_steps = 5 # Max steps to try on a single point before moving on
        
        while used < self.budget and time.time() < deadline:
            if not nd_points:
                # Random restart
                x = lo + span * np.random.rand(self.dim)
                f = problem(x)
                used += 1
                update_set(x, f)
                continue
                
            # Pick a random non-dominated point to refine
            idx = np.random.randint(len(nd_points))
            x_current = nd_points[idx]
            f_current = nd_f[idx]
            
            improved = False
            for _ in range(max_local_steps):
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Generate neighbor
                # Use a step size that scales with the remaining budget or just fixed
                step = 0.05 * span
                x_neighbor = x_current + np.random.randn(self.dim) * step
                x_neighbor = np.clip(x_neighbor, lo, hi)
                
                # Check if neighbor is likely better?
                # We can't check without evaluating.
                f_neighbor = problem(x_neighbor)
                used += 1
                
                # If neighbor is non-dominated, update current point to neighbor
                if not is_dominated(f_neighbor, nd_f):
                    # Update the set
                    # Remove current point if dominated by neighbor?
                    # Actually, we just add neighbor to the set.
                    # If neighbor dominates current, current will be removed.
                    update_set(x_neighbor, f_neighbor)
                    x_current = x_neighbor
                    f_current = f_neighbor
                    improved = True
                else:
                    # If dominated, try a smaller step or different direction?
                    # For simplicity, just break and try another parent
                    break
            
            # If no improvement in local search, try a random jump
            if not improved and np.random.rand() < 0.2:
                x_random = lo + span * np.random.rand(self.dim)
                f_random = problem(x_random)
                used += 1
                update_set(x_random, f_random)

        return None
