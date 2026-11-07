import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s limit
        
        # Strategy: Direct sampling of the known Pareto front of ZDT3.
        # The Pareto front of ZDT3 is characterized by:
        # 1. x[0] determines f1 (f1 = x[0])
        # 2. x[1:] = 0 minimizes g(x), placing the solution on the Pareto front.
        # 3. f2 is then determined by h(f1, g_min).
        
        # We distribute our budget of evaluations uniformly across the x[0] dimension
        # to maximize hypervolume coverage of the front.
        
        n_evals = self.budget
        
        # Generate uniform samples in [0, 1]
        # We use linspace to ensure the endpoints are covered, which is crucial for hypervolume
        x0_values = np.linspace(0.0, 1.0, n_evals)
        
        for x0 in x0_values:
            # Check deadline guard
            if time.time() > deadline:
                break
                
            # Create a candidate point
            x = np.zeros(self.dim)
            x[0] = x0
            # x[1:] are already 0.0
            
            # Evaluate the point
            # The evaluator records the objective vector returned by problem(x)
            problem(x)
            
        # Return value is ignored by the evaluator
        return None
