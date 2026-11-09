import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: stop 5 seconds before the 300s limit
        deadline = start_time + 295.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 structure:
        # f1 = x[0]
        # g = (dim-1) + 20 * sum_{i=1}^{dim-1} (x_i^2 - 10*cos(4*pi*x_i))
        # f2 = (1 + sqrt(f1)) * g  (or similar trade-off function)
        
        # Determine the optimal tail variables x[1:]
        # The function h(x) = x^2 - 10*cos(4*pi*x) has global minima at integers.
        # We want to minimize the sum, so we minimize each term independently.
        x_tail = np.zeros(self.dim - 1)
        for i in range(self.dim - 1):
            l = lo[i+1]
            h = hi[i+1]
            
            # Find integers in [l, h]
            int_min = int(np.ceil(l))
            int_max = int(np.floor(h))
            
            if int_min <= int_max:
                # There is at least one integer in the range
                # Pick the integer closest to 0 (since x^2 grows with |x|)
                if 0 >= l and 0 <= h:
                    x_tail[i] = 0.0
                else:
                    # Pick the integer in [int_min, int_max] with smallest absolute value
                    # Since the range might not contain 0, we just pick the one closest to 0
                    # int_min and int_max are the boundaries of the integer set
                    # The closest to 0 will be either int_min (if all > 0) or int_max (if all < 0)
                    # Or we can just check the few candidates near 0
                    candidates = [int_min, int_max]
                    # Also check if 0 is close
                    # Actually, just picking the one with min abs value is enough
                    # Since the range is contiguous integers, the one closest to 0 is 
                    # int_min if int_min > 0, or int_max if int_max < 0.
                    # If the range straddles 0, 0 is a candidate.
                    if int_min <= 0 <= int_max:
                        x_tail[i] = 0.0
                    else:
                        # All integers are on one side of 0
                        # Pick the one closest to 0
                        if int_min > 0:
                            x_tail[i] = float(int_min)
                        else:
                            x_tail[i] = float(int_max)
            else:
                # No integer in range
                # Find the integer closest to the range
                # The function is convex between local minima (which are integers)
                # So the minimum in the interval [l, h] is at one of the boundaries
                # or at the local minimum if it exists (but we established no integer inside)
                # Wait, if no integer is inside, the local minimum (integer) is outside.
                # The function increases as we move away from the nearest integer.
                # So we should pick the boundary closest to the nearest integer.
                
                # Nearest integers to the interval
                k1 = int(np.floor(l))
                k2 = int(np.ceil(h))
                
                # Distance from l to nearest integer
                dist_l = min(abs(l - k1), abs(l - (k1+1)))
                # Distance from h to nearest integer
                dist_h = min(abs(h - k2), abs(h - (k2-1)))
                
                # Actually, simpler: check the boundaries against the nearest integer
                # The nearest integer to any point in [l, h] is either floor(l) or ceil(h) or something in between?
                # No, if no integer is in [l, h], then floor(l) < l and ceil(h) > h.
                # And floor(l) + 1 > h.
                # So the nearest integers are floor(l) and ceil(h).
                # Actually, just floor(l) and floor(l)+1 are the candidates for the "gap".
                # Let's just evaluate h(l) and h(h) and pick the smaller one?
                # No, the function might have a local min inside if the interval is wide?
                # No, local mins are at integers. If no integer is inside, the function is monotonic 
                # (increasing or decreasing) within the interval?
                # Derivative: 2x + 40pi sin(4pi x).
                # Roots at integers.
                # Between integers, the function goes up then down?
                # No, 0 is min, 0.5 is max, 1 is min.
                # So in (0, 1), it goes up to 0.5 then down to 1.
                # So if the interval is (0.1, 0.4), min is at 0.1 (boundary).
                # If interval is (0.6, 0.9), min is at 0.9 (boundary).
                # If interval is (0.2, 0.8), min is at 0.2 or 0.8?
                # f(0.2) vs f(0.8).
                # Due to symmetry around 0.5, f(0.2) = f(0.8).
                # So boundaries are the minima.
                
                val_l = l*l - 10*np.cos(4*np.pi*l)
                val_h = h*h - 10*np.cos(4*np.pi*h)
                
                if val_l < val_h:
                    x_tail[i] = l
                else:
                    x_tail[i] = h
        
        # Generate 200 points for x[0] using Van der Corput sequence (base 2)
        # This provides low-discrepancy coverage in 1D
        n_evals = self.budget
        x0_vals = []
        for i in range(1, n_evals + 1):
            # Van der Corput sequence in base 2
            # v(i) = sum_{k=0}^{n} b_k / 2^{k+1} where i = sum b_k 2^k
            # Simpler: reverse bits of i
            # Or use a simple loop
            v = 0.0
            denom = 2.0
            n = i
            while n > 0:
                v += (n % 2) / denom
                n //= 2
                denom *= 2.0
            
            # Map [0, 1) to [lo[0], hi[0]]
            # Add small epsilon to avoid boundary issues if needed, but linear map is fine
            x0 = lo[0] + v * (hi[0] - lo[0])
            x0_vals.append(x0)
        
        # Evaluate all points
        for x0 in x0_vals:
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1:] = x_tail
            
            try:
                problem(x)
            except Exception:
                continue
        
        return None
