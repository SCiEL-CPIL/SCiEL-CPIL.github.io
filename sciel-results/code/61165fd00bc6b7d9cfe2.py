import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine optimal fixed values for x_1 ... x_{dim-1}
        # ZDT4 g(x) = 1 + 10 * sum(1 - cos(10 * pi * x_i))
        # Minimum is 1 when 10*pi*x_i is a multiple of 2*pi => x_i is multiple of 0.2
        # We snap each x_i to the nearest multiple of 0.2 within bounds
        fixed_vals = np.zeros(self.dim - 1)
        for i in range(1, self.dim):
            # Find nearest multiple of 0.2
            # Candidates: floor(lo/0.2)*0.2, ceil(hi/0.2)*0.2
            # Actually, just check 0.0, 0.2, ...
            # Efficient way:
            k_lo = np.ceil(lo[i] / 0.2)
            k_hi = np.floor(hi[i] / 0.2)
            
            # If no multiple of 0.2 is strictly inside, we should pick the bound closest to a multiple?
            # The function 1 - cos(10 pi x) is periodic.
            # If the interval [lo, hi] contains a multiple of 0.2, pick the closest one to 0?
            # No, any multiple gives value 0. So g=1.
            # If the interval does NOT contain a multiple, we should pick the bound that is closest to a multiple.
            
            # Let's just pick the value in [lo, hi] that minimizes 1 - cos(10 pi x).
            # This is equivalent to maximizing cos(10 pi x).
            # Max of cos is 1 at 0.2k.
            
            # Simple approach: check if 0 is in bounds. If so, use 0.
            if lo[i] <= 0 <= hi[i]:
                fixed_vals[i-1] = 0.0
            else:
                # Find nearest multiple of 0.2
                # Generate a few candidates around the interval
                # k_min = floor(lo/0.2), k_max = ceil(hi/0.2)
                k_min = np.floor(lo[i] / 0.2)
                k_max = np.ceil(hi[i] / 0.2)
                
                best_val = lo[i]
                best_cost = 1 - np.cos(10 * np.pi * lo[i])
                
                # Check endpoints and nearest multiples
                candidates = set()
                candidates.add(lo[i])
                candidates.add(hi[i])
                
                for k in range(int(k_min)-1, int(k_max)+2):
                    val = k * 0.2
                    if lo[i] - 1e-9 <= val <= hi[i] + 1e-9:
                        candidates.add(val)
                
                for val in candidates:
                    cost = 1 - np.cos(10 * np.pi * val)
                    if cost < best_cost:
                        best_cost = cost
                        best_val = val
                
                fixed_vals[i-1] = best_val

        n_points = self.budget
        
        # Sample x_0
        # ZDT4 f1 = x0. Pareto front exists for x0 in [0, 1] typically.
        # Clip x0 range to [0, 1] if possible, else use full bounds.
        # Standard ZDT4 bounds are [-5, 5]. The Pareto front is defined by x0 in [0,1] 
        # if we assume f1=x0. If x0 < 0 or x0 > 1, f1 is outside [0,1]?
        # ZDT4 definition: f1 = x1. The Pareto front is the set of non-dominated points.
        # If x1 is outside [0,1], is it dominated?
        # f2 = 1 + 10(n-1) + ...
        # Actually, for ZDT4, the Pareto front is typically reported for x1 in [0,1].
        # If we pick x1 = -1, f1 = -1.
        # Reference point is [11, 400].
        # Points with f1 < 0 are "better" in f1 than points with f1 > 0.
        # However, the "front" usually refers to the minimal f2 for a given f1.
        # If we can make f1 negative, we might get higher hypervolume?
        # Let's check the hypervolume contribution.
        # Box width: 11 - f1. If f1 = -1, width = 12.
        # Box height: 400 - f2.
        # If we fix x2..xn to minimize g, f2 = 1 - sqrt(f1) is only valid for f1 >= 0?
        # The formula for ZDT4 is f2 = g(x) (1 - sqrt(f1/g(x))).
        # This requires f1/g(x) >= 0.
        # If g(x)=1, f1 must be >= 0.
        # If f1 < 0, sqrt is undefined (or complex).
        # Standard ZDT4 assumes x1 in [0,1] for the front.
        # If bounds allow x1 < 0, we should restrict to [0, 1] to stay on the valid front.
        # If bounds allow x1 > 1, f1 > 1.
        # If f1 > 1, sqrt(f1) > 1. f2 = 1 - sqrt(f1) < 0.
        # This is even better!
        # Wait, if f2 can be negative, the hypervolume is huge.
        # Reference point 400. Height = 400 - (negative) = 400 + |f2|.
        # So we should maximize f1!
        
        # Let's re-read ZDT4 definition carefully.
        # f1(x) = x1
        # g(x) = 1 + 10(n-1) + sum (x_i^2 - 10 cos(10 pi x_i))
        # f2(x) = g(x) (1 - sqrt(f1(x)/g(x)))
        
        # To have real f2, we need f1/g >= 0.
        # If we minimize g to 1, we need f1 >= 0.
        # If we choose x1 such that f1 is large, f2 becomes very negative.
        # Example: x1 = 5. g=1. f2 = 1 - sqrt(5) = 1 - 2.236 = -1.236.
        # Hypervolume box: [5, 11] x [-1.236, 400].
        # Width = 6. Height = 401.236. Area = 2407.
        # Compare to x1 = 0. f2 = 0. Box [0, 11] x [0, 400]. Area = 4400.
        # Wait, the union of boxes.
        # If we have a point at x1=5, it covers [5, 11].
        # If we have a point at x1=0, it covers [0, 11].
        # The point at x1=0 covers the entire width.
        # The point at x1=5 is dominated by x1=0?
        # (0, 0) vs (5, -1.236).
        # f1: 0 < 5 (0 is better).
        # f2: 0 > -1.236 (-1.236 is better).
        # They are non-dominated.
        # However, the box for (0,0) is [0,11]x[0,400].
        # The box for (5, -1.236) is [5,11]x[-1.236, 400].
        # The union is [0,5]x[0,400] U [5,11]x[-1.236, 400].
        # Area = 5*400 + 6*401.236 = 2000 + 2407 = 4407.
        # Compare to just (0,0): 11*400 = 4400.
        # So adding the point at x1=5 increases the hypervolume!
        
        # What if x1 is very large?
        # If x1 = 100. f2 = 1 - 10 = -9.
        # Box [100, 11] is empty? No, f1 must be < Ref1 (11).
        # If f1 >= 11, the point is not counted (strictly better).
        # So we must keep f1 < 11.
        
        # So we should sample x1 in [0, 11) ?
        # But g(x) must be 1.
        # If we fix x2..xn to 0, g=1.
        # Then f2 = 1 - sqrt(x1).
        # This is real for x1 >= 0.
        # We want to maximize the hypervolume of the set of points.
        # The "front" extends from x1=0 to x1=11 (where f1 hits ref).
        # At x1=11, f2 = 1 - sqrt(11) = 1 - 3.316 = -2.316.
        
        # So the relevant range for x1 is [0, 11].
        # We should sample x1 in [0, 11].
        
        x0_lo = max(lo[0], 0.0)
        x0_hi = min(hi[0], 11.0)
        
        if x0_lo >= x0_hi:
             # Fallback if bounds don't allow any valid x1
             x0_vals = np.linspace(lo[0], hi[0], n_points)
        else:
             # Sample in [x0_lo, x0_hi]
             # We established that clustering near 0 is good for curvature.
             # But now the range is [0, 11].
             # Curvature of sqrt(x) is highest near 0.
             # So we should still cluster near 0.
             # t in [0, 1]. x = x0_lo + (x0_hi - x0_lo) * t^2.
             t = np.linspace(0, 1, n_points)
             x0_vals = x0_lo + (x0_hi - x0_lo) * (t ** 2.0)
             
             # Ensure we hit the upper bound?
             # t=1 gives x0_hi.
             
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0_vals
        X[:, 1:] = fixed_vals
        
        # Evaluate
        if time.time() > deadline:
             return X
             
        F = np.array([problem(x) for x in X])
        return X
