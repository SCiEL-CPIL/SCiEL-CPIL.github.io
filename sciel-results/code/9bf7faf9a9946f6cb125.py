import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325 # 5s margin
        
        best_x = None
        best_f = np.inf
        calls = 0
        dim = self.dim
        budget = self.budget

        def wrapper(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return 1e9 # Penalize
            if calls >= budget:
                return 1e9
            calls += 1
            x = np.clip(x, -5, 5)
            try:
                f = float(func(x))
            except:
                f = 1e9
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Structural Change: Multi-Start Nelder-Mead
        # Instead of a single global search (DE) followed by local refinement (NM),
        # we perform multiple independent local searches (NM) from diverse starting points.
        # This approach is more robust for multimodal functions by directly exploring
        # multiple basins of attraction, avoiding the risk of a single global search
        # failing to locate the global basin.
        
        num_restarts = 10
        # Allocate budget per restart
        # NM in 5D typically requires ~5-10 calls per iteration.
        # We allocate enough budget for ~50 calls per restart.
        budget_per_restart = max(10, budget // num_restarts)
        
        rng = np.random.RandomState(42)
        
        # Generate diverse starting points
        # Mix of center and random points to cover the domain
        starting_points = []
        starting_points.append(np.zeros(dim))
        
        for _ in range(num_restarts - 1):
            starting_points.append(rng.uniform(-5, 5, dim))
            
        # Shuffle to avoid bias in the order of exploration
        rng.shuffle(starting_points)
        
        for x0 in starting_points:
            if calls >= budget:
                break
            if time.time() > deadline - 5:
                break
                
            # Limit maxiter to stay within the allocated sub-budget
            # Assuming ~5 calls per NM iteration in 5D
            maxiter = max(1, int(budget_per_restart / 5))
            
            try:
                res = minimize(wrapper, x0, method='Nelder-Mead', 
                               options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-8})
            except:
                pass

        return best_x, best_f
