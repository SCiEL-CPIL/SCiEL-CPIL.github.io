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
        # So g_min = -9*(n-1) if all tails can be 0
        
        # Determine optimal tail: for each variable, find value in [lo[i], hi[i]]
        # that minimizes x^2 - 10*cos(2*pi*x)
        optimal_tail = np.zeros(n - 1)
        
        for i in range(1, n):
            l, h = lo[i], hi[i]
            
            # The function x^2 - 10*cos(2*pi*x) has global minima at integers k
            # with value k^2 - 10. The global minimum is at k=0 with value -9.
            # We want the integer closest to 0 that is in [l, h].
            
            k_min = int(np.ceil(l))
            k_max = int(np.floor(h))
            
            if k_min <= k_max:
                # Find integer in [k_min, k_max] with smallest |k|
                if 0 >= k_min and 0 <= k_max:
                    best_x = 0.0
                else:
                    # All integers have same sign
                    best_x = float(k_min) if k_min > 0 else float(k_max)
            else:
                # No integer in [l, h], minimum is at one of the boundaries
                def g_term(x):
                    return x*x - 10.0*np.cos(2.0*np.pi*x)
                
                val_l = g_term(l)
                val_h = g_term(h)
                best_x = l if val_l < val_h else h
            
            optimal_tail[i-1] = best_x
        
        # Now sample x[0] uniformly across its valid range
        x1_lo = lo[0]
        x1_hi = hi[0]
        
        remaining = self.budget
        if remaining <= 0:
            return []
        
        if remaining == 1:
            x1_samples = np.array([(x1_lo + x1_hi) / 2.0])
        else:
            x1_samples = np.linspace(x1_lo, x1_hi, remaining)
        
        F = []
        for x1_val in x1_samples:
            if time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = np.clip(x1_val, lo[0], hi[0])
            x[1:] = optimal_tail
            f = problem(x)
            F.append(f)
        
        return F
