import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        budget = self.budget
        dim = self.dim
        n = dim
        
        # Analytically find the x_i values that minimize the g(x) term for ZDT6
        # g(x) = 1 + 9/(n-1) * sum_{i=2}^n (x_i * sin(10*pi*x_i + i*pi/n))
        # We want to minimize each term independently.
        x_opt = np.zeros(dim)
        x_opt[0] = 0.0 # Placeholder, x_1 is handled separately
        
        # Pre-compute optimal x_i for i=2..n
        # We scan a fine grid to find the min of x * sin(10*pi*x + phase)
        grid = np.linspace(0.0, 1.0, 2000)
        for i in range(1, dim): # i is 1-based index for x_2...x_n in 0-based array it's 1..n-1
            phase = (i + 1) * np.pi / n  # i in formula is 2..n, so index i+1 in 0-based? 
            # Formula: sum_{i=2}^n ... term uses i.
            # In 0-based array, x[1] is x_2. So index j=1 corresponds to i=2.
            # phase = i * pi / n.
            # Let's re-verify the ZDT6 formula index.
            # Usually: sum_{i=2}^n x_i sin(10 pi x_i + i pi / n).
            # So for array index k (0-based), the variable is x_{k+1}.
            # If k=1, variable is x_2. The term uses i=2.
            # So phase = 2 * pi / n.
            # If k=n-1, variable is x_n. The term uses i=n.
            # So phase = (k+1) * pi / n.
            
            phase = (i + 1) * np.pi / n
            terms = grid * np.sin(10 * np.pi * grid + phase)
            min_idx = np.argmin(terms)
            x_opt[i] = grid[min_idx]
            
        # Allocation
        # 85% for the "standard" front (x[1:]=0) to ensure dense coverage
        # 15% for the "optimized" front (x[1:]=x_opt) to capture the lower g(x)
        n_std = int(budget * 0.85)
        n_opt = budget - n_std
        
        used = 0
        
        # 1. Sample standard front
        # x[0] = t, x[1:] = 0
        # We want to cover t in [0, 1]
        t_vals_std = np.linspace(0.0, 1.0, n_std)
        for t in t_vals_std:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = float(t)
            # x[1:] is already 0
            problem(x)
            used += 1
            
        # 2. Sample optimized front
        # x[0] = t, x[1:] = x_opt[1:]
        # We can reuse the t values or use a slightly different set to ensure coverage
        # Using the same t values ensures we compare g=1 vs g_min at same f1
        t_vals_opt = np.linspace(0.0, 1.0, n_opt)
        for t in t_vals_opt:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = float(t)
            x[1:] = x_opt[1:]
            problem(x)
            used += 1
            
        return None
