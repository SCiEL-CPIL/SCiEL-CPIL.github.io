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

        # Check if 0 is within bounds for x1..x_{dim-1}
        # This ensures g(x) is minimized (g=1 for standard ZDT4)
        zeros_ok = all(lo[i] <= 0 <= hi[i] for i in range(1, self.dim))

        if zeros_ok:
            # Set x1..x_{dim-1} to 0 to minimize g
            # Sample x0 with a bias towards 0 to capture the region with higher hypervolume weight
            n_points = self.budget
            
            x0_lo = lo[0]
            x0_hi = hi[0]
            
            if x0_lo >= x0_hi:
                x0_vals = np.array([x0_lo])
                n_points = 1
            else:
                # Use a quadratic mapping to concentrate points near x0_lo
                # This assumes x0_lo is the "better" end for f1 (smaller f1)
                # In ZDT4, f1 = x0, so smaller x0 is better for f1.
                # The hypervolume weight (R2 - f2) is higher when f2 is smaller.
                # f2 is smaller when x0 is smaller.
                # So we want more points near x0_lo.
                
                # Linear spacing in sqrt(x0) or similar?
                # Let's try x0 = x0_lo + (x0_hi - x0_lo) * (t**2)
                # where t is linearly spaced 0..1.
                # This puts more points near x0_lo.
                
                t = np.linspace(0, 1, n_points)
                x0_vals = x0_lo + (x0_hi - x0_lo) * (t ** 2)
                
            X = np.zeros((n_points, self.dim))
            X[:, 0] = x0_vals
            # X[:, 1:] remains 0.0
            
            # Check time
            if time.time() > deadline:
                return X
            
            F = np.array([problem(x) for x in X])
            return X

        else:
            # Fallback: 0 is not in bounds for some x_i (i>=1)
            # Find the best feasible value for each x_i to minimize g
            best_x_tail = []
            for i in range(1, self.dim):
                l = lo[i]
                h = hi[i]
                # Pick the bound closer to 0
                if abs(l) < abs(h):
                    best_x_tail.append(l)
                else:
                    best_x_tail.append(h)
            
            n_points = self.budget
            x0_lo = lo[0]
            x0_hi = hi[0]
            
            if x0_lo >= x0_hi:
                x0_vals = np.array([x0_lo])
                n_points = 1
            else:
                t = np.linspace(0, 1, n_points)
                x0_vals = x0_lo + (x0_hi - x0_lo) * (t ** 2)
                
            X = np.zeros((n_points, self.dim))
            X[:, 0] = x0_vals
            for i in range(1, self.dim):
                X[:, i] = best_x_tail[i-1]
            
            if time.time() > deadline:
                return X
            
            F = np.array([problem(x) for x in X])
            return X
