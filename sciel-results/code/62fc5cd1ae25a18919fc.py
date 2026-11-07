import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        # Reserve some budget for refinement
        n_init = int(n_evals * 0.8)
        n_refine = n_evals - n_init
        
        # 1. Initial uniform sampling of x_1
        X_init = np.zeros((n_init, self.dim))
        x1_samples = np.linspace(lo[0], hi[0], n_init)
        X_init[:, 0] = x1_samples
        X_init[:, 1:] = lo[1:]
        
        F_init = np.array([problem(x) for x in X_init])
        
        # 2. Refine the best points (lowest f_2)
        # Sort by f_2 to find the "valleys"
        # We want to find local minima of f_2(x_1)
        # Pick the top K points with lowest f_2
        K = min(10, n_refine // 2) # Refine 10 points with 2 evals each?
        
        # Actually, let's just do a simple local search on the x_1 coordinate
        # for the points with the lowest f_2 values.
        
        # Get indices of the K best points
        best_indices = np.argsort(F_init[:, 1])[:K]
        
        X_refine = []
        F_refine = []
        
        for idx in best_indices:
            x_best = X_init[idx].copy()
            f_best = F_init[idx].copy()
            
            # Try to improve x_1
            # Use a small step size
            step = 0.01
            for _ in range(n_refine // K): # Use remaining budget
                # Try random perturbations
                for _ in range(2): # 2 trials per iteration?
                    if len(X_refine) + len(F_refine) >= n_refine:
                        break
                    x_trial = x_best.copy()
                    x_trial[0] = x_best[0] + np.random.uniform(-step, step)
                    x_trial[0] = np.clip(x_trial[0], lo[0], hi[0])
                    
                    f_trial = problem(x_trial)
                    X_refine.append(x_trial)
                    F_refine.append(f_trial)
                    
                    # If better, update
                    if f_trial[1] < f_best[1]:
                        x_best = x_trial
                        f_best = f_trial
            
        return None
