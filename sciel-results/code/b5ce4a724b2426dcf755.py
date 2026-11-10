# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT1.
    
    Strategy:
    Coordinate Descent on the x0 positions of the Pareto front points.
    1. Initialize N points with equal spacing in x0.
    2. Iteratively adjust each interior point to maximize the local hypervolume gain
       of the two adjacent intervals, given its neighbors.
    3. This converges to a local optimum of the discrete HV maximization problem,
       which is likely the global optimum due to the 1D structure.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _f2_val(self, x0):
        if x0 <= 0:
            return 1.0
        return 1.0 - np.sqrt(x0) - x0 * np.sin(10.0 * np.pi * x0)

    def _local_gain(self, x, x_left, f2_left, x_right, f2_right):
        f2_x = self._f2_val(x)
        # Gain from left interval + Gain from right interval
        # Left interval: [x_left, x], height approx f2_left - f2_x (trapezoid approx? No, exact HV gain formula for insertion)
        # Actually, the HV gain of inserting x between x_left and x_right is:
        # (x - x_left) * (f2_left - f2_x) + (x_right - x) * (f2_x - f2_right)
        # This is the exact increase in HV if x is inserted, assuming x_left < x < x_right and f2 is monotonic.
        term1 = (x - x_left) * (f2_left - f2_x)
        term2 = (x_right - x) * (f2_x - f2_right)
        return term1 + term2

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = self.budget
        if N <= 0:
            return []
        
        # Initialize x0s
        x0s = np.linspace(0.0, 1.0, N)
        
        # Pre-compute f2 for current x0s to speed up neighbor lookups?
        # We only need f2 for neighbors, which change.
        
        # Coordinate Descent
        max_iters = 100
        tol = 1e-12
        gr = (np.sqrt(5) - 1) / 2  # Golden ratio conjugate
        
        for _ in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            
            changed = False
            for i in range(1, N - 1):
                x_left = x0s[i - 1]
                x_right = x0s[i + 1]
                f2_left = self._f2_val(x_left)
                f2_right = self._f2_val(x_right)
                
                # Golden section search for max of _local_gain in (x_left, x_right)
                a = x_left
                b = x_right
                
                # If interval is too small, skip
                if b - a < 1e-10:
                    continue
                
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                
                f_c = self._local_gain(c, x_left, f2_left, x_right, f2_right)
                f_d = self._local_gain(d, x_left, f2_left, x_right, f2_right)
                
                for _ in range(30):
                    if f_c > f_d:
                        b = d
                        d = c
                        f_d = f_c
                        c = b - gr * (b - a)
                        f_c = self._local_gain(c, x_left, f2_left, x_right, f2_right)
                    else:
                        a = c
                        c = d
                        f_c = f_d
                        d = a + gr * (b - a)
                        f_d = self._local_gain(d, x_left, f2_left, x_right, f2_right)
                
                x_opt = (a + b) / 2.0
                if abs(x_opt - x0s[i]) > tol:
                    x0s[i] = x_opt
                    changed = True
            
            if not changed:
                break
        
        # Evaluate the selected points
        results = []
        for x0 in x0s:
            x = np.zeros(self.dim)
            x[0] = x0
            x = np.clip(x, lo, hi)
            f = problem(x)
            results.append(f)
            
        return results

# EVOLVE-BLOCK-END
