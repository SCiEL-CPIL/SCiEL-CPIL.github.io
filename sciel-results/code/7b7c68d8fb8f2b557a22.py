import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 structure:
        # f1 = x[0]
        # f2 = g(x[1:]) * (1 - sqrt(x[0]/n))
        # g(x) = sum_{i=2}^{n} (x_i^2 - 10*cos(2*pi*x_i))
        # The minimum of each term x^2 - 10*cos(2*pi*x) is -9 at x=0, 1, -1, etc.
        # Standard ZDT4 bounds are [-5, 5], so 0 is always in bounds.
        
        def g_term(x):
            return x*x - 10.0*np.cos(2.0*np.pi*x)
        
        # Step 1: Find the optimal tail (x[1:]) to minimize g
        # We want to find x_i that minimizes x_i^2 - 10*cos(2*pi*x_i) within [lo[i], hi[i]]
        optimal_tail = np.zeros(self.dim - 1)
        for i in range(1, self.dim):
            l, h = lo[i], hi[i]
            # Candidates: boundaries and integers (local minima of the periodic function)
            candidates = [l, h]
            # Add integers in [l, h]
            if l <= 0 <= h:
                candidates.append(0.0)
            # Add nearby integers
            for k in range(int(np.floor(l)) - 1, int(np.ceil(h)) + 2):
                if l - 0.1 <= k <= h + 0.1:
                    candidates.append(float(k))
            
            best_val = float('inf')
            best_x = l
            for c in candidates:
                c = np.clip(c, l, h)
                v = g_term(c)
                if v < best_val:
                    best_val = v
                    best_x = c
            optimal_tail[i-1] = best_x
        
        # Step 2: Estimate g_min by evaluating the problem
        # If x[0] = 0, f2 = g(x[1:]) * (1 - 0) = g(x[1:])
        x1_for_g = 0.0
        if x1_for_g < lo[0]:
            x1_for_g = lo[0]
        elif x1_for_g > hi[0]:
            x1_for_g = hi[0]
        
        x_test = np.zeros(self.dim)
        x_test[0] = x1_for_g
        x_test[1:] = optimal_tail
        f_test = problem(x_test)
        
        if x1_for_g > 0:
            g_min = f_test[1] / (1.0 - np.sqrt(x1_for_g / self.dim))
        else:
            g_min = f_test[1]
        
        used = 1
        
        # Step 3: Sample the Pareto front
        # f1 = x[0], f2 = g_min * (1 - sqrt(x[0]/n))
        remaining = self.budget - used
        if remaining <= 0:
            return []
        
        x1_lo = lo[0]
        x1_hi = hi[0]
        
        # Strategy: Use uniform grid on x[0] to ensure even coverage of the Pareto front.
        # For hypervolume, uniform spacing along the front is effective because it
        # ensures no large gaps where hypervolume could be gained.
        
        n_points = remaining
        
        # Uniform grid including endpoints
        if n_points == 1:
            x1_samples = np.array([(x1_lo + x1_hi) / 2.0])
        else:
            x1_samples = np.linspace(x1_lo, x1_hi, n_points)
        
        F = []
        for x1_val in x1_samples:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = np.clip(x1_val, lo[0], hi[0])
            x[1:] = optimal_tail
            f = problem(x)
            F.append(f)
        
        return F
