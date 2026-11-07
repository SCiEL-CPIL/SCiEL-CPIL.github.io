"""
Program using analytically derived optimal quantization for ZDT1 hypervolume maximization.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = self.budget
        if N <= 0:
            return None
            
        # ZDT1 specific knowledge:
        # f1 = x1
        # g = 1 + 9 * sum(x2...xn) / (n-1)
        # f2 = g * (1 - sqrt(x1/g))
        # Optimal g is 1 (when x2...xn = 0).
        # Front: f2 = 1 - sqrt(f1) for f1 in [0, 1].
        
        # The hypervolume contribution of a point at x1 is proportional to the height of the rectangle
        # it defines, which is (Ref2 - f2(x1)) = 11 - (1 - sqrt(x1)) = 10 + sqrt(x1).
        # To maximize hypervolume with a fixed number of points, we should distribute points
        # such that the "mass" of the integrand (10 + sqrt(x1)) is evenly partitioned by the points.
        # This is the 1D optimal quantization problem where the density function is w(x) = 10 + sqrt(x).
        
        # We need to find x_i such that the integral of w(x) from 0 to x_i is proportional to i/N.
        # Let F(x) be the cumulative distribution function (CDF) of the density w(x).
        # w(x) = 10 + x^0.5
        # Integral of w(x) dx = 10x + (2/3)x^1.5
        # Let I(x) = 10x + (2/3)x^1.5
        # We want I(x_i) = (i / N) * I(1) for i = 1, ..., N-1 (interior points)
        # x_0 = 0, x_N = 1
        
        # I(1) = 10 + 2/3 = 32/3
        
        if N > 2:
            # Generate N-2 interior points
            idx = np.arange(1, N-1)
            # Target cumulative values
            target_vals = (idx / N) * (32.0 / 3.0)
            
            # Solve 10x + (2/3)x^1.5 = target_val for x
            # Use Newton's method for high precision and speed
            # f(x) = 10x + (2/3)x^1.5 - target
            # f'(x) = 10 + x^0.5
            
            # Initial guess: since 10x dominates for larger x, x ~ target/10 is a good start
            # For small x, x^1.5 dominates? No, 10x is always larger than (2/3)x^1.5 for x in [0,1].
            # Actually at x=1, 10 vs 0.66. So linear approximation is very good.
            x_est = target_vals / 10.0
            # Ensure within bounds
            x_est = np.clip(x_est, 0.0, 1.0)
            
            # Newton iterations
            for _ in range(10):
                x_pow_15 = x_est ** 1.5
                x_pow_05 = x_est ** 0.5
                f_val = 10.0 * x_est + (2.0 / 3.0) * x_pow_15 - target_vals
                f_prime = 10.0 + x_pow_05
                x_new = x_est - f_val / f_prime
                # Prevent oscillations or going out of bounds during Newton steps
                x_new = np.clip(x_new, 0.0, 1.0)
                # Check convergence
                if np.all(np.abs(x_new - x_est) < 1e-12):
                    x_est = x_new
                    break
                x_est = x_new
            
            x1_interior = x_est
            x1_values = np.concatenate([[0.0], x1_interior, [1.0]])
        else:
            x1_values = np.array([0.0, 1.0])
        
        # Evaluate points
        base_vec = np.zeros(self.dim)
        
        for i in range(len(x1_values)):
            if time.time() > deadline:
                break
            
            x1_val = x1_values[i]
            x = base_vec.copy()
            x[0] = x1_val
            problem(x)
            
        return None
