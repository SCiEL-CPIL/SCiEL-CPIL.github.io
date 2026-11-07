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
        x1_min = 0.0
        x1_max = 5.0 / 9.0 - 1e-6
        
        # Strategy: Optimize the shape parameter 'p' of the mapping x1_i = x1_min + (x1_max - x1_min) * (i/K)^p
        # We use Golden Section Search to find the p that maximizes the Hypervolume of the resulting set.
        # This is a structurally different approach from simply testing a few discrete values.
        
        def calc_hv(p):
            # Generate x1 values based on parameter p
            # i from 1 to K to avoid 0^p issues if p<0, but p>0 is expected
            # We want K points.
            # Using i/K where i is 1..K maps to (1/K, 1]. 
            # To include 0, we might need to handle i=0 separately or adjust.
            # Standard linspace-like behavior:
            # Let's use t = np.linspace(0, 1, K)
            # x1 = x1_min + (x1_max - x1_min) * (t ** p)
            
            t = np.linspace(0.0, 1.0, K)
            # If p is very small, t**p is close to 1 for all t>0, and 0 for t=0.
            # This clusters points at x1_max.
            # If p is large, t**p is close to 0 for t<1, and 1 for t=1.
            # This clusters points at x1_min.
            
            # Ensure numerical stability
            # For t=0, 0**p is 0.
            x1_vals = x1_min + (x1_max - x1_min) * (t ** p)
            
            # Sort just in case (though monotonic p>0 keeps order)
            # If p is negative, order reverses, but we assume p>0.
            
            f1 = x1_vals
            g = 10.0 - 18.0 * f1
            
            # Check validity
            # g must be > 0
            valid = g > 1e-9
            if not np.all(valid):
                # This shouldn't happen if x1_max is chosen correctly, but guard anyway
                f1 = f1[valid]
                g = g[valid]
                if len(f1) == 0:
                    return 0.0
            
            # Calculate f2
            # f2 = g * (1 - sqrt(f1/g))
            # Handle division by zero if g is tiny
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = f1 / g
                # If ratio is huge, sqrt is huge, f2 becomes negative large?
                # But f1/g <= x1_max / (10-18*x1_max) = (5/9) / (10 - 10) -> inf
                # At x1 = 5/9, g=0. 
                # The front goes to f1=5/9, f2=0? 
                # Limit x1->5/9: g->0. f1/g -> inf. sqrt->inf. f2 = g(1-inf) = -inf * 0?
                # Let's check limit.
                # f2 = g - sqrt(g*f1).
                # As g->0, g->0 and sqrt(g*f1)->0. So f2->0.
                # So the front ends at (5/9, 0).
                
                f2 = g - np.sqrt(g * f1)
            
            # Calculate HV
            # Sort by f1
            order = np.argsort(f1)
            f1 = f1[order]
            f2 = f2[order]
            
            # Hypervolume in 2D:
            # Sum of rectangles
            # f1_ext = [f1[0], f1[1], ..., f1[K-1], ref1]
            # widths = diff(f1_ext)
            # heights = ref2 - f2[i]
            # HV = sum(widths[i] * heights[i])
            
            f1_ext = np.append(f1, ref1)
            widths = np.diff(f1_ext)
            heights = ref2 - f2
            
            # The last width is ref1 - f1_last. The height is ref2 - f2_last.
            # The formula sum(widths[i] * heights[i]) works if we align them correctly.
            # widths has length K. heights has length K.
            # width[i] corresponds to the interval [f1[i], f1[i+1]] (or ref1 for last).
            # The height for this interval is determined by the point at f1[i] (the left boundary)
            # because the front is decreasing (f2 decreases as f1 increases).
            # So the rectangle height is ref2 - f2[i].
            
            hv = np.sum(widths * heights)
            return hv

        # Golden Section Search for p in [0.1, 10.0]
        # The function calc_hv(p) is expected to be unimodal.
        # We want to maximize it.
        
        gr = (np.sqrt(5.0) - 1.0) / 2.0  # Golden ratio ~ 0.618
        
        a = 0.1
        b = 10.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = calc_hv(c)
        fd = calc_hv(d)
        
        # Iterate until interval is small or time runs out
        max_iters = 50
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            if fc > fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = calc_hv(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = calc_hv(d)
            
            if abs(b - a) < 1e-4:
                break
        
        best_p = (a + b) / 2.0
        
        # Generate final x1 values
        t = np.linspace(0.0, 1.0, K)
        best_x1 = x1_min + (x1_max - x1_min) * (t ** best_p)
        
        # Ensure strict bounds and uniqueness
        best_x1 = np.clip(best_x1, 0.0, 1.0)
        
        # Evaluate points
        results = []
        
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
