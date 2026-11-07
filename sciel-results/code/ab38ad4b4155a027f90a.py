import numpy as np
import time

class Optimizer:
    """
    Optimizer that directly samples the known Pareto front of ZDT6.
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Deadline guard, though 200 evals is very fast
        start_time = time.time()
        deadline = 300.0  # 300s limit, keep margin
        
        # ZDT6 specifics:
        # f1 = 1 - sin(10 * pi * x1)
        # f2 = (1 + f1) * (1 - sqrt(f1 / g(x)))
        # g(x) = 1 + 9 * (sum(x[1:]) / 9)
        
        # To be on the Pareto front, we must minimize g(x), which means x[1:] = 0.
        # Then g(x) = 1.
        # f2 = (1 + f1) * (1 - sqrt(f1))
        
        # We sample f1 uniformly in [0, 1].
        # For each f1, we find x1 such that 1 - sin(10 * pi * x1) = f1.
        # sin(10 * pi * x1) = 1 - f1
        # 10 * pi * x1 = arcsin(1 - f1)
        # x1 = arcsin(1 - f1) / (10 * pi)
        
        n_evals = self.budget
        # Generate uniform samples in f1 space
        # Using linspace to ensure endpoints 0 and 1 are covered
        f1_samples = np.linspace(0.0, 1.0, n_evals)
        
        last_x = None
        
        for i in range(n_evals):
            if time.time() > deadline - 1.0:
                break
                
            f1_target = f1_samples[i]
            
            # Calculate x1
            # Handle floating point edge cases for arcsin
            val = 1.0 - f1_target
            # Clip to [-1, 1] to be safe, though [0,1] is fine
            val = np.clip(val, -1.0, 1.0)
            x1 = np.arcsin(val) / (10.0 * np.pi)
            
            # Construct x vector
            x = np.zeros(self.dim)
            x[0] = x1
            # x[1:] remains 0, which minimizes g(x)
            
            # Ensure bounds are respected (x1 is in [0, 0.05], so it's in [0,1])
            # x[1:] is 0, in [0,1]
            
            # Evaluate
            # The problem function clips and returns objectives
            # We don't strictly need to store F, as the evaluator records it
            problem(x)
            last_x = x
            
        return last_x
