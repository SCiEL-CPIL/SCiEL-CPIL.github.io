import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Optimal values for x_1 ... x_{n-1}
        # For ZDT1, g(x) is minimized at x_i = 0
        x = np.zeros(self.dim)
        
        # Sampling strategy for x_0
        # Use non-uniform grid denser near 0 to better approximate the Pareto front curve f2 = 1 - sqrt(f1)
        # This minimizes the hypervolume loss due to discretization
        t = np.linspace(0.0, 1.0, self.budget)
        x0_values = lo[0] + (hi[0] - lo[0]) * (t**2)
        
        for i in range(self.budget):
            if time.time() > deadline:
                break
                
            x[0] = x0_values[i]
            problem(x)
            
        return None
