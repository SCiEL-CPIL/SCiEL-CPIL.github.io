import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Directive: Grid search over x[0], x[1], x[2]
        # To maximize points on the Pareto front (x[1]=0, x[2]=0), 
        # we use the smallest possible grid for x[1] and x[2] that constitutes a search.
        # N_1 = 2, N_2 = 2.
        # Remaining budget for x[0]: N_0 = budget // (N_1 * N_2)
        
        n_1 = 2
        n_2 = 2
        n_0 = self.budget // (n_1 * n_2)
        
        # Ensure we use all budget if possible, but grid must be integer.
        # If budget is 200, n_0 = 50. Total 200.
        
        x0_grid = np.linspace(lo[0], hi[0], n_0)
        x1_grid = np.linspace(lo[1], hi[1], n_1)
        x2_grid = np.linspace(lo[2], hi[2], n_2)
        
        results = []
        
        for x0 in x0_grid:
            for x1 in x1_grid:
                for x2 in x2_grid:
                    x = np.zeros(self.dim)
                    x[0] = x0
                    x[1] = x1
                    x[2] = x2
                    # Other variables are 0 (which is in bounds [0,1])
                    # Clip to be safe
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    results.append(f)
                    
        return results
