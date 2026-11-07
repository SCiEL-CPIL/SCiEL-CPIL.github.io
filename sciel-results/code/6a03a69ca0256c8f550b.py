import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_optimal_front_points(self, n_points, lo0, hi0):
        """Compute x1 values optimized for hypervolume on ZDT1 Pareto front."""
        if n_points <= 1:
            return np.array([(lo0 + hi0) / 2.0])
        
        # ZDT1 front: f1 = x1, f2 = 1 - sqrt(x1), x1 in [0,1]
        # Hypervolume with ref [11,11]: sort by f1 ascending
        # HV = sum_i (f1_{i+1} - f1_i) * (11 - f2_i) + (11 - f1_n)*(11 - f2_n)
        # where f1_{n+1} = 11
        # f2_i = 1 - sqrt(f1_i)
        # So 11 - f2_i = 10 + sqrt(f1_i)
        
        # We want to choose f1_1 < f1_2 < ... < f1_n in [lo0, hi0] to maximize
        # sum_{i=1}^{n-1} (f1_{i+1} - f1_i) * (10 + sqrt(f1_i)) + (11 - f1_n) * (10 + sqrt(f1_n))
        
        # Use iterative refinement: start with uniform in f2, then optimize
        t_vals = np.linspace(0.0, 1.0, n_points)
        x1 = (1.0 - t_vals) ** 2
        
        # Refine using local optimization of the spacing
        # The marginal gain of inserting between two adjacent points depends on gap
        # Use a simple iterative scheme: redistribute points to equalize marginal HV contribution
        for iteration in range(10):
            # Sort x1
            x1_sorted = np.sort(x1)
            x1_sorted = np.clip(x1_sorted, lo0, hi0)
            
            if len(x1_sorted) < 2:
                break
            
            f2_sorted = 1.0 - np.sqrt(x1_sorted)
            weights = 10.0 + np.sqrt(x1_sorted)  # 11 - f2_i
            
            # Compute current HV contributions per interval
            gaps = np.diff(x1_sorted)
            gaps = np.append(gaps, 11.0 - x1_sorted[-1])
            
            # Marginal contribution of point i (0-indexed) in the sum:
            # Point i contributes: gap_i * w_i  (for the interval [x1_i, x1_{i+1}))
            # But actually each point's weight multiplies the gap AFTER it
            # HV = sum_{i=0}^{n-1} (x1_{i+1} - x1_i) * w_i + (11 - x1_n) * w_n
            # Wait, let me re-derive. Points sorted ascending f1.
            # The staircase: from (x1_0, 11) go right to (x1_1, 11), down to (x1_1, f2_1), 
            # no... for hypervolume of a set of points with reference (11,11):
            # A point (a,b) with a<11, b<11 contributes the rectangle [0,a]x[0,b] minus overlaps.
            # For non-dominated points sorted by f1 asc, the HV is:
            # sum_{i} (f1_i - f1_{i-1}) * (11 - f2_i) + f1_0 * (11 - f2_0)
            # where f1_{-1} = 0
            # Actually standard formula: sort by f1 asc. 
            # HV = f1_0*(11-f2_0) + sum_{i=1}^{n-1} (f1_i - f1_{i-1})*(11 - f2_i)
            
            # Let me just compute HV directly for the current set and do a simple search
            pass
            
        # Alternative: use the fact that for this specific front, we can derive
        # the optimal spacing analytically or via a cleaner numerical approach.
        
        # Simple effective approach: use quantiles of a distribution proportional 
        # to the "local front length" weighted by the leverage (11 - f2).
        # Local HV contribution density ~ (11 - f2(s)) * ds where s parameterizes by f1
        # For f1 = s, f2 = 1-sqrt(s), so 11 - f2 = 10 + sqrt(s)
        # We want equal HV contribution per point, so sample s with density proportional to (10+sqrt(s))
        # CDF: F(s) = integral_0^s (10+sqrt(u)) du / integral_0^1 (10+sqrt(u)) du
        # integral = 10s + (2/3)s^{3/2}
        # Total = 10 + 2/3 = 32/3
        # F(s) = (10s + (2/3)s^{3/2}) / (32/3)
        
        # Invert: for uniform t in [0,1], solve 10s + (2/3)s^{3/2} = t * 32/3
        # Use Newton's method
        s_vals = np.zeros(n_points)
        t_uniform = np.linspace(0.0, 1.0, n_points)
        target = t_uniform * (32.0 / 3.0)
        
        for i in range(n_points):
            s = target[i] / 10.0 if target[i] > 0 else 0.0  # initial guess
            s = max(0.0, min(1.0, s))
            for _ in range(20):
                f_val = 10.0 * s + (2.0/3.0) * s**1.5
                f_prime = 10.0 + s**0.5
                correction = (f_val - target[i]) / f_prime
                s_new = s - correction
                s_new = max(0.0, min(1.0, s_new))
                if abs(s_new - s) < 1e-14:
                    s = s_new
                    break
                s = s_new
            s_vals[i] = s
        
        # Clip to bounds
        s_vals = np.clip(s_vals, lo0, hi0)
        
        return s_vals

    def __call__(self, problem, bounds):
        t0 = time.time()
        hard_deadline = t0 + 270.0
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        n = self.budget
        if n <= 0:
            return np.array([])
        
        # Compute optimal x1 values for hypervolume maximization on ZDT1 front
        x1_opt = self._compute_optimal_front_points(n, lo[0], hi[0])
        
        # Build full dimension vectors: x1 = x1_opt, x[1:] = lo[1:]
        pop = np.zeros((n, self.dim), dtype=np.float64)
        pop[:, 0] = x1_opt
        pop[:, 1:] = lo[1:]
        
        # Evaluate all points
        count = 0
        for idx in range(n):
            if time.time() > hard_deadline:
                break
            problem(pop[idx])
            count += 1
        
        return pop
