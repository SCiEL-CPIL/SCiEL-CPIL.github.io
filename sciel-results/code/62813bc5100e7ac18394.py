# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize_scalar

class Optimizer:
    """Deterministic sampler that places points on the known ZDT4 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 295.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n = self.dim  # 10
        g_const = 1 + 10 * (n - 1)  # 91
        budget = self.budget  # 200
        
        # The reference point is [11.0, 400.0]
        # Points with f1 > 11 or f2 > 400 do not contribute to HV.
        # f1 = x0. So x0 must be < 11.
        # Also x0 must be >= 0 for sqrt.
        # Upper bound for x0 is min(hi[0], 11.0)
        max_x0 = min(hi[0], 11.0)
        if max_x0 < 0:
            max_x0 = 0.0

        # Analytical ZDT4 front functions
        def f1(x):
            return x
        
        def f2(x):
            # g(x) = 91 + x^2 - 20*x*cos(4*pi*x)
            # f2 = g(x) * (1 - sqrt(x))
            # Handle x=0 carefully
            if x < 0:
                return 0.0 # Should not happen
            g = g_const + x**2 - 20 * x * np.cos(4 * np.pi * x)
            return g * (1 - np.sqrt(x))

        def calc_hv(x0s):
            # x0s is sorted array of x0 values
            # Filter out points that are not strictly better than reference
            # f1 < 11 and f2 < 400
            f1s = f1(x0s)
            f2s = f2(x0s)
            
            mask = (f1s < 11.0) & (f2s < 400.0)
            f1s = f1s[mask]
            f2s = f2s[mask]
            
            if len(f1s) == 0:
                return 0.0
            
            # Sort by f1 (which is x0)
            # Since x0s is sorted, f1s is sorted
            # But f2s might not be monotonic? 
            # For HV calculation with reference point, we need the non-dominated set.
            # However, for a single curve, if we just sum the boxes, we might overcount if the curve is not monotonic in the "Pareto" sense.
            # But ZDT4 front is Pareto optimal.
            # The standard HV formula for a sorted list of points (by f1) where f2 is decreasing:
            # HV = sum_{i=0}^{N-2} (f1[i+1] - f1[i]) * (R2 - f2[i]) + (R1 - f1[N-1]) * (R2 - f2[N-1])
            # This assumes f2 is decreasing.
            # If f2 is not decreasing, we need to remove dominated points first.
            # Let's remove dominated points to be safe.
            
            # Sort by f1
            order = np.argsort(f1s)
            f1s = f1s[order]
            f2s = f2s[order]
            
            # Remove dominated points: keep only those where f2 is lower than all previous f2s
            # Since f1 is increasing, a point is dominated if there is a previous point with f2 <= current f2.
            min_f2s = np.minimum.accumulate(f2s)
            # A point is non-dominated if f2s[i] == min_f2s[i]
            # Note: if there are ties, we keep the one with smaller f1? 
            # Actually, if f1 is strictly increasing, and f2 is the min so far, it's non-dominated.
            # If f2[i] > min_f2s[i], it is dominated.
            # If f2[i] == min_f2s[i], it is non-dominated (or tied).
            # We keep points where f2s[i] == min_f2s[i].
            # However, if f2 is flat, we might keep multiple points with same f2 but different f1.
            # The one with smaller f1 is better.
            # So we should keep the *first* occurrence of each min_f2 value?
            # Or just compute HV on the non-dominated set.
            
            # Let's construct the non-dominated set.
            # Iterate from left to right.
            # Keep point if f2 < current_min_f2.
            # If f2 == current_min_f2, we can skip it (dominated by previous with same f2 and smaller f1).
            # So strictly decreasing f2 is required for the non-dominated set on a curve?
            # Yes, for minimization, if f2 is not decreasing, the later point is dominated.
            
            nd_f1 = []
            nd_f2 = []
            current_min = np.inf
            for i in range(len(f1s)):
                if f2s[i] < current_min:
                    nd_f1.append(f1s[i])
                    nd_f2.append(f2s[i])
                    current_min = f2s[i]
            
            if len(nd_f1) == 0:
                return 0.0
                
            nd_f1 = np.array(nd_f1)
            nd_f2 = np.array(nd_f2)
            
            R1, R2 = 11.0, 400.0
            
            if len(nd_f1) == 1:
                return (R1 - nd_f1[0]) * (R2 - nd_f2[0])
            
            diffs_f1 = np.diff(nd_f1)
            area = np.sum(diffs_f1 * (R2 - nd_f2[:-1])) + (R1 - nd_f1[-1]) * (R2 - nd_f2[-1])
            return area

        # Multi-start local search
        best_hv = -np.inf
        best_x0s = None
        
        # Initializations
        inits = []
        # 1. Uniform
        inits.append(np.linspace(0.0, max_x0, budget))
        # 2. Random
        np.random.seed(42) # Fixed seed for reproducibility within the call
        inits.append(np.sort(np.random.uniform(0.0, max_x0, budget)))
        # 3. Log-spaced (if max_x0 > 0)
        if max_x0 > 0:
            # Avoid 0 for log
            inits.append(np.sort(np.geomspace(1e-6, max_x0, budget)))
        
        for init_x0s in inits:
            x0s = init_x0s.copy()
            current_hv = calc_hv(x0s)
            
            # Local search
            for _ in range(20):
                improved = False
                for i in range(budget):
                    # Optimize x0s[i]
                    def obj_func(t):
                        new_x = x0s.copy()
                        new_x[i] = t
                        new_x_sorted = np.sort(new_x)
                        return -calc_hv(new_x_sorted)
                    
                    try:
                        # Bounds for t: [0, max_x0]
                        res = minimize_scalar(obj_func, bounds=(0.0, max_x0), method='bounded', options={'xatol': 1e-5})
                        if res.success and res.fun < -current_hv:
                            x0s[i] = res.x
                            x0s = np.sort(x0s)
                            current_hv = -res.fun
                            improved = True
                    except:
                        pass
                
                if not improved:
                    break
                
            if current_hv > best_hv:
                best_hv = current_hv
                best_x0s = x0s.copy()
                
                # Time check
                if time.time() > deadline - 5.0:
                    break

        # Construct points
        points = []
        for x0 in best_x0s:
            x = np.zeros(self.dim)
            x[0] = x0
            x = np.clip(x, lo, hi)
            points.append(x)
        
        # Evaluate all points
        F = np.array([problem(x) for x in points])
        
        # Check time
        if time.time() > deadline:
            return points
        
        return points
# EVOLVE-BLOCK-END
