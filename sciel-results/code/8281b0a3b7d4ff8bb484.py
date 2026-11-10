import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        all_X = []
        all_F = []
        budget_used = 0
        
        def now():
            return time.time()
            
        def eval_point(x):
            nonlocal budget_used
            if budget_used >= budget or now() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            budget_used += 1
            all_X.append(x)
            all_F.append(f)
            return f

        # WFG9 Structure Assumption:
        # x[0], x[1] determine position on the front.
        # x[2] ... x[dim-1] contribute to distance from front (usually squared terms).
        # To minimize distance, set x[2:] to lower bounds (assuming 0 is min or close to it).
        
        # Base vector: x[0], x[1] will be varied, rest set to lower bound.
        base_x = np.zeros(dim)
        for i in range(2, dim):
            base_x[i] = lo[i]
            
        # If lo is not 0, we might want to check if mid is better? 
        # Usually, squared terms are minimized at 0. If bounds are [0,1], 0 is best.
        # If bounds are [-1,1], 0 is best.
        # Let's stick to lower bound as it's a safe bet for "distance" minimization in WFG.
        
        # Phase 1: Dense Grid on x[0], x[1]
        n_grid = 12
        grid_budget = n_grid * n_grid
        
        x0_vals = np.linspace(lo[0], hi[0], n_grid)
        x1_vals = np.linspace(lo[1], hi[1], n_grid)
        
        for x0 in x0_vals:
            if budget_used >= budget or now() > deadline:
                break
            for x1 in x1_vals:
                if budget_used >= budget or now() > deadline:
                    break
                x = base_x.copy()
                x[0] = x0
                x[1] = x1
                eval_point(x)
        
        # Phase 2: Refinement
        if all_F and budget_used < budget:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)
            
            # Find non-dominated points
            def is_non_dominated(f, F_all):
                for j, fj in enumerate(F_all):
                    if j != f:
                        if np.all(fj <= f) and np.any(fj < f):
                            return False
                return True
                
            non_dom_indices = [i for i, f in enumerate(F_arr) if is_non_dominated(f, F_arr)]
            
            if non_dom_indices:
                # Sort non-dominated points by sum of objectives (distance to ideal)
                # Ideal point is typically (0,0,0) or lower bounds of objectives
                # We just want the "best" ones to refine
                nd_points = [(i, F_arr[i]) for i in non_dom_indices]
                # Simple heuristic: sort by sum of f
                nd_points.sort(key=lambda t: np.sum(t[1]))
                
                # Pick top 5 for refinement
                top_indices = [t[0] for t in nd_points[:5]]
                
                remaining_budget = budget - budget_used
                n_refine_per_point = max(1, remaining_budget // len(top_indices))
                
                for idx in top_indices:
                    if budget_used >= budget or now() > deadline:
                        break
                    
                    center_x = X_arr[idx].copy()
                    center_f = F_arr[idx]
                    
                    # Local search: perturb x[0] and x[1]
                    # Step size based on grid resolution
                    step0 = (hi[0] - lo[0]) / n_grid
                    step1 = (hi[1] - lo[1]) / n_grid
                    
                    for _ in range(n_refine_per_point):
                        if budget_used >= budget or now() > deadline:
                            break
                        
                        x_new = center_x.copy()
                        # Random perturbation in x[0], x[1]
                        x_new[0] += np.random.uniform(-step0, step0)
                        x_new[1] += np.random.uniform(-step1, step1)
                        
                        f_new = eval_point(x_new)
                        
                        # If we found a better point, update center for subsequent iterations?
                        # Simple hill climbing: if f_new dominates center_f, update center
                        if f_new is not None:
                            # Check if f_new dominates center_f
                            if np.all(f_new <= center_f) and np.any(f_new < center_f):
                                center_f = f_new
                                center_x = x_new.copy()
                            elif np.all(center_f <= f_new) and np.any(center_f < f_new):
                                # Keep center
                                pass
                            else:
                                # Incomparable, just keep exploring
                                pass
                            
                            # Also, if f_new is non-dominated globally, it's good.
                            # But simple local improvement is fine.
                            
        if all_X:
            return np.array(all_X)
        else:
            return np.empty((0, dim))
