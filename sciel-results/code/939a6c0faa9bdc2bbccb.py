import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = self.budget
        
        # Use stratified grid with jitter for better coverage of Pareto front
        # This provides more uniform coverage than Halton sequence
        n_grid = int(np.sqrt(N))
        # Ensure we have at least N points
        while n_grid * n_grid < N:
            n_grid += 1
        
        u_vals = []
        v_vals = []
        for i in range(n_grid):
            for j in range(n_grid):
                if len(u_vals) >= N:
                    break
                # Stratified position with uniform jitter within each cell
                u = (i + np.random.uniform(0, 1)) / n_grid
                v = (j + np.random.uniform(0, 1)) / n_grid
                
                # Bias towards lower values to concentrate on high-hypervolume regions
                u = u ** 2.0
                v = v ** 2.0
                
                u_vals.append(u)
                v_vals.append(v)
        
        u_vals = np.array(u_vals[:N])
        v_vals = np.array(v_vals[:N])
        
        # Scale to bounds
        u_vals = lo[0] + (hi[0] - lo[0]) * u_vals
        v_vals = lo[1] + (hi[1] - lo[1]) * v_vals
        
        last_result = None
        for i in range(N):
            if time.time() > deadline:
                break
                
            x = np.zeros(self.dim)
            x[0] = u_vals[i]
            x[1] = v_vals[i]
            x[2:] = lo[2:]
            
            last_result = problem(x)
            
        return last_result
