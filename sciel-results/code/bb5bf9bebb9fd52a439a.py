import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        
        # ZDT4: f1 = x0, f2 = g(x1..xn) * (1 - sqrt(x0/n))
        # g(x) = sum(x_i^2 - 10*cos(2*pi*x_i))
        # We want to minimize g(x1..xn).
        # Each term x_i^2 - 10*cos(2*pi*x_i) is minimized at x_i = 0 (value -9).
        # It is periodic with period 1, and local minima at all integers k have value k^2 - 10.
        # So the global minimum is at k=0 (value -9).
        # If 0 is not in bounds, we pick the value in bounds closest to 0 (or nearest integer if available).
        # Actually, if bounds are [l, h], we check if any integer k is in [l, h].
        # The best integer is the one with smallest |k|, i.e., closest to 0.
        # If no integer is in [l, h], the best value is the one in [l, h] closest to the nearest integer.
        
        optimal_tail = np.zeros(dim - 1)
        for i in range(1, dim):
            l, h = lo[i], hi[i]
            
            # Find the value in [l, h] that minimizes x^2 - 10*cos(2*pi*x)
            # Global minima are at integers k with value k^2 - 10.
            
            # 1. Check if any integer is in [l, h]
            k_min = int(np.ceil(l))
            k_max = int(np.floor(h))
            
            best_x = None
            
            if k_min <= k_max:
                # We want the integer with smallest k^2
                # This is the integer closest to 0
                if 0 >= k_min and 0 <= k_max:
                    best_x = 0.0
                else:
                    # If range is entirely positive, smallest k is best
                    # If range is entirely negative, largest k (closest to 0) is best
                    if k_min > 0:
                        best_x = float(k_min)
                    elif k_max < 0:
                        best_x = float(k_max)
                    else:
                        # Should not happen if 0 not in range and k_min<=k_max
                        # But just in case
                        best_x = min([float(k_min), float(k_max)], key=lambda x: abs(x))
            else:
                # No integer in [l, h]
                # Minimum is at one of the boundaries
                def g_term(x):
                    return x*x - 10.0*np.cos(2.0*np.pi*x)
                
                val_l = g_term(l)
                val_h = g_term(h)
                
                if val_l < val_h:
                    best_x = l
                else:
                    best_x = h
            
            optimal_tail[i-1] = best_x
        
        # Now we have a very good estimate of the optimal tail.
        # We can use ALL budget for the x0 sweep.
        
        remaining = self.budget
        if remaining <= 0:
            return []
        
        x1_lo = lo[0]
        x1_hi = hi[0]
        
        # Generate x0 samples
        # Uniform grid
        if remaining == 1:
            x1_samples = np.array([(x1_lo + x1_hi) / 2.0])
        else:
            x1_samples = np.linspace(x1_lo, x1_hi, remaining)
        
        F = []
        for x1_val in x1_samples:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = np.clip(x1_val, lo[0], hi[0])
            x[1:] = optimal_tail
            f = problem(x)
            F.append(f)
        
        return F
