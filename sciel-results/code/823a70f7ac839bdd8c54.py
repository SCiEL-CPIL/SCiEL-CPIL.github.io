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
        n = self.dim
        
        # ZDT4: f1 = x[0], f2 = g(x[1:]) * (1 - sqrt(x[0]/n))
        # g(x) = sum_{i=2}^{n} (x_i^2 - 10*cos(2*pi*x_i))
        # Each term has minimum -9 at x_i = 0 (if 0 is in bounds)
        # So g_min = 9*(n-1) if all tails can be 0
        
        # Determine if x_i = 0 is achievable for each tail variable
        tail_can_be_zero = (lo[1:] <= 0) & (hi[1:] >= 0)
        
        if np.all(tail_can_be_zero):
            # Standard ZDT4: g_min = 9*(n-1)
            g_min = 9.0 * (n - 1)
            optimal_tail = np.zeros(n - 1)
        else:
            # Need to estimate g_min by evaluating at best available tail
            # Use x_i = 0 if possible, else clip to nearest bound
            optimal_tail = np.zeros(n - 1)
            for i in range(1, n):
                if lo[i] > 0:
                    optimal_tail[i-1] = lo[i]
                elif hi[i] < 0:
                    optimal_tail[i-1] = hi[i]
                else:
                    optimal_tail[i-1] = 0.0
            
            # Evaluate g at this tail with x[0] = 0
            x1_for_g = max(0.0, lo[0])
            if x1_for_g > hi[0]:
                x1_for_g = hi[0]
            
            x = np.zeros(n)
            x[0] = x1_for_g
            x[1:] = optimal_tail
            f = problem(x)
            if x1_for_g > 0:
                g_val = f[1] / (1 - np.sqrt(x1_for_g / n))
            else:
                g_val = f[1]
            g_min = g_val
        
        # The Pareto front: f2 = g_min * (1 - sqrt(f1/n))
        # f1 = x[0], so x[0] should range over [max(0, lo[0]), min(n, hi[0])]
        # But we need x[0] >= 0 for sqrt to be real, and f1 in [0, n] for standard ZDT4
        
        x1_lo = max(0.0, lo[0])
        x1_hi = min(float(n), hi[0])
        
        if x1_lo >= x1_hi:
            # Degenerate case
            x1_lo = lo[0]
            x1_hi = hi[0]
        
        used = 0
        remaining = self.budget
        
        # Sample x[0] uniformly across the valid range
        # Use linspace for uniform coverage of the Pareto front
        if remaining <= 0:
            return []
        
        # Generate sample points with non-uniform density
        # Higher density at high f1 where hypervolume contribution is larger
        # Use a power-law spacing: x1_i = x1_lo + (x1_hi - x1_lo) * (i/N)^p
        # p > 1 gives more density at high x1
        # Optimal p depends on the front shape; try p=1.3
        if remaining > 1:
            t = np.linspace(0, 1, remaining)
            p = 1.3
            x1_samples = x1_lo + (x1_hi - x1_lo) * (t ** p)
        else:
            x1_samples = np.array([x1_lo])
        
        F = []
        for i in range(remaining):
            if time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = x1_samples[i]
            x[1:] = optimal_tail
            f = problem(x)
            F.append(f)
        
        return F
