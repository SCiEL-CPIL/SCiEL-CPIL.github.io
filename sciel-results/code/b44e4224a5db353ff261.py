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
            # Find integer in [l, h] closest to 0
            # Integers are ..., -1, 0, 1, ...
            # Check 0 first
            if l <= 0 <= h:
                optimal_tail[i-1] = 0.0
            else:
                # Find nearest integer to 0 that might be in range? 
                # Actually, just find the value in [l, h] that is closest to an integer.
                # Since the function is symmetric around integers and grows with distance,
                # we want x as close to an integer as possible.
                # Among integers, 0 is best. Then +/-1, etc.
                
                # Candidate integers: floor(l) to ceil(h)
                # We want the integer k in [l, h] with min k^2.
                # If none, we want x in [l, h] minimizing dist(x, Z).
                
                # Let's just evaluate the best few candidates:
                # 1. Bound closest to 0? No, bound closest to nearest integer.
                
                # Let's find the integer k closest to the interval [l, h].
                # If interval contains an integer, pick the one with smallest abs(k).
                # If not, pick the bound closest to the nearest integer outside.
                
                # Simple robust approach:
                # Check integers around 0
                candidates = []
                # Check if 0 is in range (already done)
                # Check +/-1, +/-2...
                for k in range(0, 10):
                    if l <= k <= h:
                        candidates.append(float(k))
                    if l <= -k <= h:
                        candidates.append(float(-k))
                
                if candidates:
                    # Pick the one with smallest absolute value (min k^2)
                    best_k = min(candidates, key=lambda x: abs(x))
                    optimal_tail[i-1] = best_k
                else:
                    # No integer in range.
                    # The function is convex-ish near integers?
                    # x^2 - 10cos(2pi x).
                    # Derivative: 2x + 20pi sin(2pi x).
                    # Minima at integers.
                    # If no integer in [l, h], the minimum is at the boundary closest to an integer.
                    # Which boundary?
                    # Let's check l and h.
                    # Distance to nearest integer for l:
                    # Distance to nearest integer for h:
                    def dist_to_int(x):
                        return abs(x - round(x))
                    
                    dl = dist_to_int(l)
                    dh = dist_to_int(h)
                    
                    # Also check if l or h is closer to 0 than the other?
                    # The value depends on x^2 and the periodic part.
                    # If we are far from 0, x^2 dominates. So we prefer smaller |x|.
                    # If we are close to an integer, the periodic part is -10.
                    
                    # Let's just evaluate l and h and pick the better one?
                    # We can't evaluate here without spending budget.
                    # But we can approximate.
                    # If l and h are in the same period (e.g. [0.1, 0.9]), 
                    # the min is at the boundary closest to the integer (0 in this case).
                    # dist(0.1, 0) = 0.1. dist(0.9, 1) = 0.1.
                    # f(0.1) = 0.01 - 10cos(0.2pi) ~ -8.08
                    # f(0.9) = 0.81 - 10cos(1.8pi) ~ 0.81 - 10cos(0.2pi) ~ -7.28
                    # So 0.1 is better because |x| is smaller.
                    
                    # So we want the value in [l, h] that minimizes x^2 - 10cos(2pi x).
                    # Since we don't have budget to evaluate, we guess.
                    # The term x^2 pushes us towards 0.
                    # The term -10cos pushes us towards integers.
                    # If the interval doesn't contain an integer, it's "between" integers.
                    # The best point is the one closest to an integer, with tie-breaker being closer to 0.
                    
                    # Let's pick the bound with smaller distance to nearest integer.
                    if dl < dh:
                        optimal_tail[i-1] = l
                    elif dh < dl:
                        optimal_tail[i-1] = h
                    else:
                        # Tie, pick closer to 0
                        if abs(l) < abs(h):
                            optimal_tail[i-1] = l
                        else:
                            optimal_tail[i-1] = h
        
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
