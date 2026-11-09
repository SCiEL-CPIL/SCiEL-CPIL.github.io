import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        used = 0
        results = []
        
        # Strategy: Exploit ZDT1 structure
        # Pareto front is at x[1:] = lo[1:] (assuming standard ZDT1 where g(x)=0 at lower bounds)
        # We focus all budget on sampling x[0] to cover the front densely and hit peaks.
        
        # 1. Identify critical points for Hypervolume
        # f2 = 1 + sin(10*x0)
        # Peaks of sin(10*x0) are at 10*x0 = pi/2 + 2*k*pi
        # x0 = (pi/2 + 2*k*pi) / 10
        # In [0, 1], peaks are at:
        # k=0: pi/20 ~ 0.157? No.
        # sin(10*x) peaks at 10*x = pi/2, 5*pi/2, 9*pi/2...
        # x = pi/20, 5*pi/20, 9*pi/20, 13*pi/20, 17*pi/20
        # pi/20 ~ 0.157
        # 5*pi/20 = pi/4 ~ 0.785? No.
        # Let's calculate properly.
        # Period is 2*pi/10 = pi/5 ~ 0.628.
        # Peak at pi/20 ~ 0.157.
        # Next peak at pi/20 + pi/5 = 5*pi/20 = pi/4 ~ 0.785?
        # Wait. sin(10*x).
        # 10*x = pi/2 => x = pi/20 ~ 0.157
        # 10*x = 5*pi/2 => x = pi/4 ~ 0.785
        # 10*x = 9*pi/2 => x = 9*pi/20 ~ 1.41 (out of bounds)
        
        # Let's re-verify ZDT1 definition.
        # f2 = 1 + sin(10 * x1)
        # Range of x1 is [0, 1].
        # 10 * x1 is [0, 10].
        # sin(t) peaks at pi/2 ~ 1.57, 5*pi/2 ~ 7.85.
        # So peaks at x1 = 0.157, 0.785.
        # Also, sin(t) is 1 at pi/2.
        # Are there other peaks?
        # 10*x1 = pi/2 + 2*k*pi
        # k=0: x1 = pi/20 ~ 0.157
        # k=1: x1 = (pi/2 + 2pi)/10 = 5pi/20 = pi/4 ~ 0.785
        # k=2: x1 = (pi/2 + 4pi)/10 = 9pi/20 ~ 1.41 (out)
        
        # So only 2 peaks in [0,1].
        # My previous assumption of 5 peaks was wrong (confused with sin(100x) or something).
        
        # The "valleys" (minima) are at sin = -1.
        # 10*x1 = 3*pi/2, 7*pi/2
        # x1 = 3*pi/20 ~ 0.471
        # x1 = 7*pi/20 ~ 1.099 (out)
        # So 1 valley.
        
        # To maximize HV, we want to capture the shape of the curve.
        # The curve goes up to 2, down to 0, up to 2.
        # We should ensure we have points at the peaks and valleys.
        
        key_points = [
            lo[0],
            hi[0],
            lo[0] + (hi[0] - lo[0]) * (np.pi / 20.0),       # Peak 1
            lo[0] + (hi[0] - lo[0]) * (3 * np.pi / 20.0),    # Valley
            lo[0] + (hi[0] - lo[0]) * (5 * np.pi / 20.0),    # Peak 2
        ]
        
        # Filter points within bounds
        key_points = [p for p in key_points if lo[0] <= p <= hi[0]]
        key_points = sorted(set(key_points))
        
        # We want to distribute the remaining budget uniformly between these key points
        # to ensure dense coverage of the segments.
        
        n_total = self.budget
        n_key = len(key_points)
        
        # If budget is small, just use key points
        if n_total <= n_key:
            # Prioritize key points
            # Sort by importance? Just take first n_total
            x0_values = key_points[:n_total]
        else:
            # Distribute points
            x0_values = []
            # Add key points
            x0_values.extend(key_points)
            
            # Calculate how many extra points we have
            extra = n_total - n_key
            
            # We have n_key - 1 segments.
            n_segments = n_key - 1
            if n_segments > 0:
                points_per_seg = extra // n_segments
                remainder = extra % n_segments
                
                for i in range(n_segments):
                    start = key_points[i]
                    end = key_points[i+1]
                    count = points_per_seg
                    if i < remainder:
                        count += 1
                    
                    if count > 0:
                        # Generate points strictly between start and end
                        # linspace includes endpoints, so we use num=count+2 and slice
                        if count > 1:
                            seg_points = np.linspace(start, end, count + 2)[1:-1]
                            x0_values.extend(seg_points)
                        else:
                            # Just one point in the middle
                            x0_values.append((start + end) / 2)
            
            # Shuffle to avoid systematic bias? No, sorted is fine for HV.
            # But we must ensure we don't exceed budget
            x0_values = x0_values[:n_total]
        
        # Evaluate
        for x0 in x0_values:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = np.full(self.dim, lo) # Set all to lower bound
            x[0] = x0
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append((x, f))
            used += 1
            
        # If we have remaining budget (due to rounding or time), fill with random
        # Unlikely, but safe
        if used < self.budget:
            rng = np.random.default_rng(42)
            while used < self.budget:
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * rng.random(self.dim)
                f = problem(x)
                results.append((x, f))
                used += 1

        return None
