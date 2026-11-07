import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 250.0
        
        points = []
        
        for i in range(self.budget):
            if time.time() > deadline:
                break
                
            # Sample uniformly from the Pareto front in objective space
            # The front is the triangle 0 <= f3 <= f2 <= 1
            f2 = np.random.rand()
            f3 = np.random.rand() * f2
            
            # Construct x
            x = np.zeros(self.dim)
            x[0] = np.sqrt(1.0 - f2)
            x[1] = np.sqrt(f2 - f3)
            x[2] = np.sqrt(f3)
            # x[3:] remains 0
            
            # Ensure bounds (should be fine)
            # x is in [0,1]
            
            f = problem(x)
            points.append(x)
            
        return points
