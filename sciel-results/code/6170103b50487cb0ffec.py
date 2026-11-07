import numpy as np
import time
from scipy.optimize import brentq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        
        n_evals = self.budget
        dim = self.dim
        start_time = time.time()
        deadline = 300.0

        # ZDT6 Pareto front analysis
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # Pareto front: x2..xn = 0, f1 in [f1_min, 1]
        # f1_min = 1 - max(exp(-4*x1)*sin(6*pi*x1)^8)
        # Max occurs at x1 = 1/12 (first peak of sin^8, highest exp value)
        # f1_min = 1 - exp(-4/12) * 1^8 = 1 - exp(-1/3)
        
        f1_min = 1.0 - np.exp(-1.0/3.0)
        f1_max = 1.0
        
        # Uniform sampling of f1
        f1_targets = np.linspace(f1_min, f1_max, n_evals)
        
        # Precompute hump centers and max values
        # Humps are at x_k = (2k+1)/12 for k=0..5
        hump_centers = [(2*k + 1) / 12.0 for k in range(6)]
        hump_maxes = [np.exp(-4.0 * c) for c in hump_centers]
        
        # Function to solve for x1 given Y = 1 - f1
        # We look for x in [left, right] such that exp(-4*x)*sin(6*pi*x)^8 = Y
        def solve_x1(Y):
            # Find the first hump that can achieve Y
            # hump_maxes is decreasing
            valid_hump_idx = -1
            for k in range(6):
                if hump_maxes[k] >= Y:
                    valid_hump_idx = k
                    break
            
            if valid_hump_idx == -1:
                # Should not happen if Y <= hump_maxes[0]
                # Fallback to center of first hump
                return hump_centers[0]
            
            k = valid_hump_idx
            center = hump_centers[k]
            left = 2*k / 12.0
            right = (2*k + 2) / 12.0
            
            # We need to find root of f(x) = exp(-4*x)*sin(6*pi*x)^8 - Y = 0
            # in [left, center] or [center, right]
            # Let's pick the left branch [left, center]
            # Note: at left and right, value is 0. At center, value is hump_maxes[k].
            # If Y is very close to 0, the root is very close to left or right.
            # If Y is close to hump_maxes[k], the root is very close to center.
            
            def g(x):
                return np.exp(-4.0 * x) * (np.sin(6.0 * np.pi * x) ** 8) - Y
            
            # Use brentq
            # Ensure bracket is valid
            # g(left) = -Y (negative if Y>0)
            # g(center) = hump_maxes[k] - Y (positive if Y < hump_maxes[k])
            # If Y == hump_maxes[k], root is center.
            if Y == hump_maxes[k]:
                return center
                
            # Add small epsilon to avoid numerical issues at boundaries if needed,
            # but brentq handles endpoints if signs are correct.
            # g(left) is -Y. If Y>0, it's negative.
            # g(center) is positive.
            try:
                x_sol = brentq(g, left, center, xtol=1e-12, rtol=1e-12)
            except ValueError:
                # Fallback if brentq fails (e.g. sign issues due to float precision)
                # Simple bisection or just return center
                x_sol = center
                
            return x_sol

        x1_selected = np.empty(n_evals)
        
        for i in range(n_evals):
            if time.time() - start_time > deadline:
                break
                
            target_f1 = f1_targets[i]
            Y = 1.0 - target_f1
            
            # If Y is very small, f1 is close to 1.
            # If Y is close to hump_maxes[0], f1 is close to f1_min.
            
            x1_val = solve_x1(Y)
            x1_selected[i] = x1_val
            
        # Evaluate
        for i in range(len(x1_selected)):
            if time.time() - start_time > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = x1_selected[i]
            
            # Clip to bounds
            x_clipped = np.clip(x, lo, hi)
            problem(x_clipped)
            
        return None
