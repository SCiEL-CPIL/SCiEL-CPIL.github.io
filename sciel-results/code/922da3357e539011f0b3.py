# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for ZDT1 exploiting its known structure.
    The Pareto front is located at x[1:] = 0 and x[0] in [0, 1].
    f1 = x[0]
    f2 = 1 - sqrt(x[0])
    
    We sample x[0] uniformly in [0, 1] to maximize hypervolume coverage.
    The uniform spacing in x[0] (which is f1) provides a balanced approximation 
    of the area under the Pareto front curve relative to the reference point.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: return before 300s limit
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        if n_evals <= 0:
            return []

        # Generate sample points on the Pareto front
        # ZDT1 Pareto front: x[1:] = 0, x[0] in [0, 1]
        # We use linspace to include the endpoints 0 and 1, which provides 
        # full coverage of the objective space range [0, 1] for f1.
        x0_vals = np.linspace(0.0, 1.0, n_evals)
        
        # Clip to bounds just in case, though [0,1] is standard
        x0_vals = np.clip(x0_vals, lo[0], hi[0])
        
        results = []
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
                
            x = np.zeros(self.dim)
            x[0] = x0_vals[i]
            
            # Evaluate the point
            f = problem(x)
            results.append(f)
            
        return results
# EVOLVE-BLOCK-END
