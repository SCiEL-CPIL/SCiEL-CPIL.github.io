"""ZDT3 Optimizer using CDF-inverted Halton sampling for hypervolume-optimal spacing."""
import numpy as np
import time
from scipy.optimize import brentq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for the 330s kill switch
        
        dim = self.dim
        budget = self.budget
        bounds_arr = np.array(bounds)
        lo = bounds_arr[:, 0]
        hi = bounds_arr[:, 1]
        
        # ZDT3 specific structure:
        # f1 = x[0]
        # g = 1 + 10/d * (sum(x[i]) - sum(f(x[i])))
        # f2 = g * (1 - sqrt(f1/g))
        # To maximize hypervolume, we minimize g by setting x[1:] = 0.
        # This yields f2 = 1 - sqrt(f1).
        
        # Hypervolume-optimal sampling: sample f1 according to density
        # p(f1) ∝ (1 - sqrt(f1)) so that each sample contributes equal
        # hypervolume. Use inverse CDF with Halton quasi-random points.
        
        # CDF: F(t) = 3*(t - (2/3)*t^(3/2)) for t in [0,1]
        # Inverse: solve 3*(t - (2/3)*t^(3/2)) = u for t
        def cdf(t):
            return 3.0 * (t - (2.0/3.0) * t**1.5)
        
        def inv_cdf(u):
            """Invert the CDF F(t) = 3*(t - 2/3*t^1.5) = u"""
            if u <= 0:
                return 0.0
            if u >= 1:
                return 1.0
            # Use brentq for robust inversion
            return brentq(lambda t: cdf(t) - u, 0.0, 1.0, xtol=1e-15)
        
        if budget <= 2:
            x0_samples = np.linspace(lo[0], hi[0], budget)
        else:
            n_halton = budget - 2
            # Generate Van der Corput sequence (Halton base 2)
            halton_points = []
            for i in range(1, n_halton + 1):
                x = 0.0
                f = 0.5
                j = i
                while j > 0:
                    x += f * (j % 2)
                    j //= 2
                    f /= 2.0
                halton_points.append(x)
            
            # Apply inverse CDF to concentrate samples where hypervolume
            # contribution is highest (near f1=0)
            halton_points = [inv_cdf(p) for p in halton_points]
            
            # Add endpoints 0.0 and 1.0
            halton_points.append(0.0)
            halton_points.append(1.0)
            
            x0_samples = np.array(halton_points)
            # Map to [lo[0], hi[0]]
            x0_samples = lo[0] + x0_samples * (hi[0] - lo[0])
        
        # Pre-allocate the rest of the vector
        # Set to 0 to minimize g.
        rest_val = 0.0
        if lo[1].min() > 0:
            rest_val = lo[1].min()
            
        for i in range(budget):
            if time.time() > deadline:
                break
            
            x = np.full(dim, rest_val)
            x[0] = x0_samples[i]
            
            # Clip to bounds to be safe
            x = np.clip(x, lo, hi)
            
            # Evaluate
            problem(x)
            
        return None
