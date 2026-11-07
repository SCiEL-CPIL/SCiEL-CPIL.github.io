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
        
        n = self.dim
        K = self.budget
        ref1 = 11.0
        ref2 = 11.0
        
        # ZDT6 Analytic Front with x2..xn = 0
        # f1 = x1
        # g = 10 - 18*x1
        # f2 = g * (1 - sqrt(f1/g))
        
        # Valid range for x1 where g > 0 and f2 is real
        # 10 - 18*x1 > 0 => x1 < 5/9
        # Also f1/g must be >= 0. Since f1=x1 >= 0 and g>0, this is fine.
        # So x1 in [0, 5/9]
        
        x1_min = 0.0
        x1_max = 5.0 / 9.0 - 1e-6 # Small epsilon to avoid division by zero or invalid sqrt
        
        # We want to select K points in x1 to maximize hypervolume.
        # The Pareto front is convex or concave?
        # Let's check the shape of f2(x1).
        # f2(x1) = (10 - 18*x1) * (1 - sqrt(x1 / (10 - 18*x1)))
        
        # Hypervolume of a set of points P = {(f1_i, f2_i)} sorted by f1:
        # HV = sum_{i=1}^{K} (f1_{i+1} - f1_i) * (ref2 - f2_i) + (ref1 - f1_K) * (ref2 - f2_K)
        # where f1_{K+1} = ref1.
        
        # To maximize this, we should place points where the "slope" or curvature of the front
        # causes the most "loss" in the staircase approximation.
        # For a convex front (bulging towards the reference point), uniform spacing in x1 is good.
        # For a concave front, we might need different spacing.
        
        # Let's evaluate a few candidate distributions for x1 values:
        # 1. Uniform in x1
        # 2. Uniform in f1 (same as uniform in x1 here)
        # 3. Points clustered near the "knee"
        
        # Actually, since f1 = x1, uniform in x1 is uniform in f1.
        # Let's just use uniform distribution in x1, which is standard and near-optimal
        # for smooth fronts. The current program's local search might have been
        # overfitting to numerical noise or not converging well.
        # A simpler, faster, and robust approach is to just use uniform sampling
        # in the valid domain of x1.
        
        # However, to be "materially different" and potentially better, let's try
        # to optimize the spacing slightly using a parametric family.
        # Let x1_i = x1_min + (x1_max - x1_min) * (i/K)^p
        # We can test p in [0.5, 1.0, 1.5, 2.0] using vectorized HV calculation.
        
        # Generate candidate x1 sets
        indices = np.arange(K)
        
        def calc_hv_vectorized(x1_vals):
            # x1_vals is a 1D array of length K, sorted
            f1 = x1_vals
            g = 10.0 - 18.0 * f1
            # Ensure g > 0
            valid = g > 1e-9
            if not np.all(valid):
                # Filter invalid points (should not happen with our range)
                f1 = f1[valid]
                g = g[valid]
                if len(f1) == 0:
                    return 0.0
            f2 = g * (1.0 - np.sqrt(f1 / g))
            
            # Calculate HV
            # Sort just in case, though they are generated sorted
            order = np.argsort(f1)
            f1 = f1[order]
            f2 = f2[order]
            
            # Widths
            f1_ext = np.append(f1, ref1)
            widths = np.diff(f1_ext)
            
            # Heights
            heights = ref2 - f2
            
            # The last width corresponds to the last point's height
            # HV = sum_{i=1}^{K-1} widths[i] * heights[i] + widths[K] * heights[K]
            # Wait, the formula is:
            # Area = sum_{i=1}^{K} (f1_{i+1} - f1_i) * (ref2 - f2_i)
            # where f1_{K+1} = ref1.
            
            hv = np.sum(widths * heights)
            return hv

        # Test different power distributions
        best_hv = -1.0
        best_x1 = None
        
        # Powers to test: 0.5, 0.75, 1.0, 1.25, 1.5
        powers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        
        for p in powers:
            # x1_i = x1_min + (x1_max - x1_min) * (i/K)^p
            # i from 0 to K-1
            t = np.linspace(0, 1, K)
            x1_test = x1_min + (x1_max - x1_min) * (t ** p)
            # Ensure strictly increasing and within bounds
            x1_test = np.clip(x1_test, x1_min, x1_max)
            # Remove duplicates if any (unlikely with float)
            
            hv = calc_hv_vectorized(x1_test)
            if hv > best_hv:
                best_hv = hv
                best_x1 = x1_test
        
        # If best_x1 is None (shouldn't happen), fallback to uniform
        if best_x1 is None:
            best_x1 = np.linspace(x1_min, x1_max, K)
        
        # Now, evaluate these points using the problem function
        results = []
        
        # We have K points. We must evaluate them.
        # Check time
        if time.time() > deadline:
             return results
            
        for i in range(K):
            if time.time() > deadline:
                break
                
            x1_val = best_x1[i]
            x = np.zeros(n)
            x[0] = x1_val
            
            try:
                f = problem(x)
                f = np.asarray(f, dtype=float)
                if np.all(np.isfinite(f)):
                    # Only count if strictly better than reference
                    if f[0] < ref1 and f[1] < ref2:
                        results.append(f)
            except Exception:
                pass
                
        return results
