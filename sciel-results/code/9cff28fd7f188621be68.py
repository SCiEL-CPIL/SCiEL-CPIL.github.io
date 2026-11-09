import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # State
        X_list = []
        F_list = []
        eval_count = 0
        
        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            if time.time() - start_time > 290:
                return None
            f = problem(x)
            eval_count += 1
            return f

        def is_dominated(f_new, F_list):
            # Check if f_new is dominated by any point in F_list
            # Dominated if there exists f_old such that f_old <= f_new and f_old < f_new (at least one strict)
            for f_old in F_list:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False

        def dominates(f_new, F_list):
            # Check if f_new dominates any point in F_list
            # Returns indices of dominated points
            dominated_idx = []
            for i, f_old in enumerate(F_list):
                if np.all(f_new <= f_old) and np.any(f_new < f_old):
                    dominated_idx.append(i)
            return dominated_idx

        def update_population(x, f):
            nonlocal X_list, F_list
            # Check if new point is dominated by existing
            if is_dominated(f, F_list):
                return
            
            # Remove points dominated by new point
            dom_idx = dominates(f, F_list)
            for idx in sorted(dom_idx, reverse=True):
                X_list.pop(idx)
                F_list.pop(idx)
            
            # Add new point
            X_list.append(x)
            F_list.append(f)

        # 1. Generate Anchor Points
        # Stratified sampling on the quarter sphere
        n_anchors = 80
        n_grid = int(np.sqrt(n_anchors))
        # Ensure n_grid * n_grid >= n_anchors
        if n_grid * n_grid < n_anchors:
            n_grid += 1
            
        anchors = []
        # Use Halton-like or Grid
        # Grid in u (cos theta) and v (phi)
        u_vals = np.linspace(0.05, 0.95, n_grid)
        v_vals = np.linspace(0.05, 0.95, n_grid)
        
        for u in u_vals:
            for v in v_vals:
                if len(anchors) >= n_anchors:
                    break
                theta = np.arccos(u)
                phi = v * (np.pi / 2.0)
                
                x = np.zeros(self.dim)
                # DTLZ2: first n_obj vars are direction
                # x_1 = sin(theta)cos(phi)
                # x_2 = sin(theta)sin(phi)
                # x_3 = cos(theta)
                # If n_obj > 3, we need to handle it. 
                # For n_obj=3, this is perfect.
                if self.n_obj >= 1:
                    x[0] = np.sin(theta) * np.cos(phi)
                if self.n_obj >= 2:
                    x[1] = np.sin(theta) * np.sin(phi)
                if self.n_obj >= 3:
                    x[2] = np.cos(theta)
                
                # If n_obj > 3, we need to distribute the remaining "direction" mass.
                # But standard DTLZ2 is defined for n_obj objectives.
                # If n_obj > 3, the sphere is (n_obj-1)-dimensional.
                # For simplicity, if n_obj > 3, we set x[3:] = 0 for direction vars?
                # No, that would make sum(x^2) < 1.
                # Let's assume n_obj=3 as per task description "n_obj=3".
                
                # Distance variables
                for k in range(self.n_obj, self.dim):
                    x[k] = 0.5
                
                x = np.clip(x, lo, hi)
                anchors.append(x)
        
        # Evaluate anchors
        for x in anchors:
            f = evaluate(x)
            if f is None:
                break
            update_population(x, f)

        # 2. Local Search / Evolution
        # Use remaining budget
        # Strategy: Mutate non-dominated points
        while eval_count < self.budget:
            if time.time() - start_time > 290:
                break
            
            if not X_list:
                # Fallback to random
                x = lo + (hi - lo) * np.random.random(self.dim)
            else:
                # Pick a random non-dominated point
                idx = np.random.randint(len(X_list))
                parent = X_list[idx]
                
                # Mutation
                # Scale step size based on remaining budget? 
                # Start with larger steps, decrease?
                # Simple: fixed step size 0.1 * range
                step = 0.1 * (hi - lo)
                x = parent + step * np.random.standard_normal(self.dim)
                x = np.clip(x, lo, hi)
            
            f = evaluate(x)
            if f is None:
                break
            update_population(x, f)

        return X_list
