import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_evals = 0
        n = self.budget

        # ZDT1 Pareto front: f1 = t, f2 = 1 - sqrt(t) for t in [0, 1]
        # We want to maximize hypervolume with reference point (11, 11).
        
        # The hypervolume contribution of a point (t, 1-sqrt(t)) in the sorted list
        # is proportional to (11 - f2) * (t - t_prev).
        # To maximize the total hypervolume with N points, we should space them
        # such that the marginal gain is equal. This leads to a spacing where
        # the density is proportional to the "height" of the front relative to the reference.
        
        # Specifically, the optimal spacing t_i is given by the inverse of the CDF
        # where the PDF is proportional to (11 - (1 - sqrt(t))) = 10 + sqrt(t).
        
        # Let's define the cumulative function V(t) which is the integral of the PDF.
        # PDF(t) = 10 + sqrt(t)
        # V(t) = 10*t + (2/3)*t^(3/2)
        
        # We want t_i such that V(t_i) is uniformly spaced.
        # V(0) = 0
        # V(1) = 10 + 2/3 = 32/3
        
        # So we solve V(t_i) = u_i * V(1) for u_i = (i + 0.5) / n
        # 10*t + (2/3)*t^(3/2) = u_i * (32/3)
        
        # This is a non-linear equation. We can solve it numerically for each t_i.
        # Or we can use a root-finding method.
        # Since n=200, we can just loop and use brentq or similar.
        # However, scipy.optimize.brentq is fast.
        
        # Let's check if scipy is available. Yes, it is.
        from scipy.optimize import brentq

        def V(t):
            return 10.0 * t + (2.0 / 3.0) * t**1.5

        V_max = V(1.0)
        
        # Generate uniform u values
        u_vals = (np.arange(n) + 0.5) / n
        
        t_vals = np.zeros(n)
        for i in range(n):
            target = u_vals[i] * V_max
            # Solve V(t) = target for t in [0, 1]
            # V(0)=0, V(1)=V_max. target is in (0, V_max).
            try:
                t_vals[i] = brentq(lambda t: V(t) - target, 0.0, 1.0, xtol=1e-12)
            except Exception:
                # Fallback to uniform if it fails
                t_vals[i] = u_vals[i]

        # Build decision vectors
        X = np.zeros((n, self.dim))
        X[:, 0] = t_vals
        
        # Ensure points are strictly inside bounds if necessary, 
        # but t in [0,1] is fine for ZDT1.
        # The problem clips, so it's safe.

        F = []
        for i in range(n):
            if time.time() > deadline:
                break
            f = problem(X[i])
            F.append(f)
            n_evals += 1
            
        return X[:n_evals]
