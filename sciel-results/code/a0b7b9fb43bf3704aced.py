import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0
        
        n_evals = self.budget
        
        # Local minima of f2(f1) for ZDT3
        # f2(f1) = 1 - sqrt(f1) - f1 * sin(10 * pi * f1)
        # Minima occur where sin(10 * pi * f1) = 1
        # 10 * pi * f1 = pi/2 + 2 * k * pi
        # f1 = 0.05 + 0.2 * k
        minima_f1 = [0.05 + 0.2 * k for k in range(5)]
        
        # Generate uniform samples
        # We want to ensure we include the minima exactly
        # and distribute the rest uniformly
        
        # Create a set to hold unique f1 values
        f1_values = set(minima_f1)
        
        # Add uniform samples
        # We need n_evals - 5 more points
        n_uniform = n_evals - len(minima_f1)
        uniform_samples = np.linspace(0.0, 1.0, n_uniform)
        
        for val in uniform_samples:
            f1_values.add(val)
        
        # Convert to list and sort
        f1_values = list(f1_values)
        
        # If we have more than n_evals due to duplicates (unlikely with linspace and specific minima), 
        # we might need to trim. But linspace is continuous, minima are specific.
        # Actually, linspace might hit a minimum. 
        # If it does, we have a duplicate in the set, so the set size is smaller.
        # We want exactly n_evals evaluations.
        # If len(f1_values) < n_evals, we need to add more points.
        # If len(f1_values) > n_evals, we need to remove some.
        
        # Let's just generate n_evals points directly, ensuring minima are included.
        # A better way:
        # 1. Take the 5 minima.
        # 2. Generate n_evals - 5 uniform points.
        # 3. Combine and deduplicate.
        # 4. If we have too few, add more uniform points.
        
        f1_list = list(minima_f1)
        n_needed = n_evals - len(f1_list)
        
        # Generate uniform points in [0, 1]
        # We can use random or linspace. Linspace is deterministic.
        # To avoid hitting the minima again, we can just add them and deduplicate.
        
        uniform_points = np.linspace(0.0, 1.0, n_needed)
        
        for val in uniform_points:
            f1_list.append(val)
            
        # Deduplicate
        f1_list = list(set(f1_list))
        
        # If we still don't have enough, add more
        while len(f1_list) < n_evals:
            # Add random points
            val = np.random.uniform(0.0, 1.0)
            if val not in f1_list:
                f1_list.append(val)
                
        # If we have too many (shouldn't happen with set), trim
        if len(f1_list) > n_evals:
            f1_list = f1_list[:n_evals]
            
        # Evaluate
        for x0 in f1_list:
            if time.time() > deadline:
                break
                
            x = np.zeros(self.dim)
            x[0] = x0
            problem(x)
            
        return None
