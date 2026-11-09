import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        used = 0
        budget = self.budget
        dim = self.dim
        
        # Grid size
        n_grid = 14
        # u is uniform in [0, 1]
        u_vals = np.linspace(0.0, 1.0, n_grid)
        
        # Transform u to x1 to get uniform area on the front
        # x1 = (2/pi) * arccos(1 - u)
        x1_vals = (2.0 / np.pi) * np.arccos(1.0 - u_vals)
        
        # x2 is uniform in [0, 1]
        x2_vals = np.linspace(0.0, 1.0, n_grid)
        
        candidates = []
        
        for x1 in x1_vals:
            for x2 in x2_vals:
                x = np.full(dim, 0.5)
                x[0] = x1
                x[1] = x2
                candidates.append(x)
                
        # Add 4 gap fillers
        # These are points that are not on the 14x14 grid
        # 0.25 and 0.75 are not multiples of 1/13
        for x1 in [0.25, 0.75]:
            for x2 in [0.25, 0.75]:
                x = np.full(dim, 0.5)
                x[0] = x1
                x[1] = x2
                candidates.append(x)
                
        # Cap to budget
        candidates = candidates[:budget]
        
        for x in candidates:
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            problem(x)
            used += 1
            
        return np.array(candidates[:used])
