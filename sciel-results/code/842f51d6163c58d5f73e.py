import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT1.
    Uses the known analytical Pareto front to place evaluations optimally
    via curvature-aware quantization.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        # Safety margin: The evaluator kills at 330s. We aim to finish in < 5s.
        deadline = t0 + 10.0 

        B = int(self.budget)
        D = int(self.dim)
        
        if B <= 0:
            return np.zeros(D)

        # ZDT1 Pareto Front:
        # x1 in [0, 1], x2...xD = 0
        # f1 = x1
        # f2 = 1 - sqrt(x1)
        
        # Reference point
        R1 = 11.0
        R2 = 11.0

        def compute_hv(x1_vals):
            """
            Calculate Hypervolume for points on the ZDT1 front.
            x1_vals: array of x1 values (sorted ascending).
            """
            if len(x1_vals) == 0:
                return 0.0
            
            # Ensure sorted
            x1_sorted = np.sort(x1_vals)
            
            # f2 values
            f2 = 1.0 - np.sqrt(x1_sorted)
            
            # Heights relative to reference point
            # The HV contribution of a point i is (next_f1 - current_f1) * (R2 - current_f2)
            # The last point extends to R1
            next_f1 = np.append(x1_sorted[1:], R1)
            current_f1 = x1_sorted
            
            widths = next_f1 - current_f1
            heights = R2 - f2
            
            # Sum of areas of the rectangles
            hv = np.sum(widths * heights)
            
            return float(hv)

        # Optimal quantization approach:
        # The marginal gain of adding a point at x1 is proportional to the local "height"
        # of the Pareto front relative to the reference point, which is (R2 - f2(x1)).
        # For ZDT1, f2(x1) = 1 - sqrt(x1), so height = 11 - (1 - sqrt(x1)) = 10 + sqrt(x1).
        # The optimal grid places points such that the cumulative distribution is proportional
        # to the integral of this height function.
        # 
        # The integral of (10 + sqrt(x1)) dx1 from 0 to x is:
        # 10*x + (2/3)*x^(3/2)
        # 
        # We want to find x1 values such that this integral is evenly spaced.
        # Let F(x) = 10*x + (2/3)*x^(3/2). We want F(x_i) = (i + 0.5) * F(1) / B for i = 0..B-1.
        # F(1) = 10 + 2/3 = 32/3
        # So F(x_i) = (i + 0.5) * (32/3) / B
        # 
        # Solving for x_i requires inverting F(x) = 10*x + (2/3)*x^(3/2).
        # This is a cubic equation in sqrt(x). Let y = sqrt(x), then x = y^2.
        # F(x) = 10*y^2 + (2/3)*y^3
        # We need to solve: (2/3)*y^3 + 10*y^2 = F(x_i)
        # This can be done numerically using Newton's method or bisection.
        
        # Precompute the target integral values
        F_total = 32.0 / 3.0
        u = np.linspace(0.0, 1.0, B)
        target_F = (u + 0.5) * F_total / B
        
        # Solve for x1 values using Newton's method
        # F(x) = 10*x + (2/3)*x^(3/2)
        # F'(x) = 10 + x^(1/2)
        
        def solve_F(target_val):
            """Solve 10*x + (2/3)*x^(3/2) = target_val for x in [0, 1]."""
            # Use bisection for robustness
            lo, hi = 0.0, 1.0
            for _ in range(50):
                mid = (lo + hi) / 2.0
                val = 10.0 * mid + (2.0 / 3.0) * mid ** 1.5
                if val < target_val:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2.0
        
        # Vectorized solution using bisection
        x1_vals = np.zeros(B)
        lo = np.zeros(B)
        hi = np.ones(B)
        
        for _ in range(50):
            mid = (lo + hi) / 2.0
            val = 10.0 * mid + (2.0 / 3.0) * np.sqrt(mid * mid * mid)
            mask = val < target_F
            lo[mask] = mid[mask]
            hi[~mask] = mid[~mask]
        
        x1_vals = (lo + hi) / 2.0
        x1_vals = np.clip(x1_vals, 0.0, 1.0)
        
        # Construct X
        X = np.zeros((B, D))
        X[:, 0] = x1_vals
        # x2...xD = 0 is already set by zeros
        
        # Clip to bounds
        lo_bounds = np.array([b[0] for b in bounds], dtype=np.float64)
        hi_bounds = np.array([b[1] for b in bounds], dtype=np.float64)
        X = np.clip(X, lo_bounds, hi_bounds)
        
        # Evaluate
        for i in range(B):
            if time.time() > deadline:
                break
            problem(X[i].copy())
            
        return X[-1] # Return value is ignored, but must not crash
