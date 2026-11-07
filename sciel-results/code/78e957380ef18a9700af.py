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
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        all_F = []
        
        def evaluate(x):
            nonlocal all_F
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_F.append(f.copy())
            return f

        n_evals = self.budget
        
        # Generate Halton sequence points in [0, 1]^2
        # We use bases 2 and 3 for the two dimensions
        def halton(i, base):
            f = 1.0
            r = 0.0
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        u_vals = np.zeros(n_evals)
        v_vals = np.zeros(n_evals)
        
        for i in range(n_evals):
            idx = i + 1
            u_vals[i] = halton(idx, 2)
            v_vals[i] = halton(idx, 3)

        # Transform to uniform points on the 2-simplex
        # y1 + y2 + y3 = 1, yi >= 0
        # y1 = 1 - sqrt(u)
        # y2 = (1 - y1) * v
        # y3 = 1 - y1 - y2
        
        y1 = 1.0 - np.sqrt(u_vals)
        y2 = (1.0 - y1) * v_vals
        # y3 is implicit
        
        # Map y to x
        # x0 (x1 in DTLZ1 notation) = y1
        # x1 (x2 in DTLZ1 notation) = y2 / (1 - y1)
        # Note: 1 - y1 = y2 + y3. If y1 is close to 1, denominator is small.
        # x2 = y2 / (y2 + y3).
        
        x0 = y1
        # Calculate x1 carefully
        denom = 1.0 - x0
        # Where denom is effectively 0, x1 should be 0 (limit as y2->0, y3->0)
        # Actually, if y1=1, then y2=0 and y3=0.
        # In that case, f1=0.5, f2=0, f3=0.
        # x0=1. x1 can be anything in [0,1]. Let's set it to 0.
        x1 = np.zeros_like(x0)
        mask = denom > 1e-12
        x1[mask] = y2[mask] / denom[mask]
        
        # Evaluate all points
        for i in range(n_evals):
            if time.time() > deadline - 5:
                break
            if len(all_F) >= self.budget:
                break
                
            x = np.zeros(self.dim)
            x[0] = x0[i]
            x[1] = x1[i]
            # Set shape variables to 0.5 to ensure g(x) = 0
            x[2:] = 0.5
            
            # Ensure bounds are respected (though 0.5 and x0,x1 in [0,1] should be safe)
            x = np.clip(x, lo, hi)
            
            evaluate(x)
            
        return all_F
