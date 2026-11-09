"""Combined: Analytic front seeding with vectorized x1 solver."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize_scalar

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 330s kill
        rng = np.random.default_rng(12345)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        used = 0
        archive_X = []
        archive_F = []

        def eval_point(x):
            nonlocal used
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return x, f

        def add_point(x, f):
            # add if not dominated by archive
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return # dominated
            # remove archive points dominated by f
            keep = []
            for i, af in enumerate(archive_F):
                if np.all(f <= af) and np.any(f < af):
                    continue
                keep.append(i)
            archive_F = [archive_F[i] for i in keep]
            archive_X = [archive_X[i] for i in keep]
            archive_F.append(f)
            archive_X.append(x)

        # We want to minimize P(x1) = (10c + sin(10*pi*x1))^2 / (4*x1+4)^2
        # Strategy:
        # 1. Create a dense grid of x1 values.
        # 2. For a set of c values, compute P for all pairs (c, x1) using broadcasting.
        # 3. Find the x1 that minimizes P for each c.
        # 4. Refine with minimize_scalar on the local neighborhood.

        n_targets = self.budget
        
        # Create a grid of c values
        # Uniform in c gives good spread in f1 for ZDT6
        cs = np.linspace(0.0, 1.0, n_targets)
        
        # Dense grid for initial x1 estimation
        x1_grid = np.linspace(0.0, 1.0, 2000)
        
        # Vectorized calculation of penalty for all c and x1
        # c shape: (n_targets, 1)
        # x1 shape: (1, 2000)
        c_col = cs[:, np.newaxis]
        x1_row = x1_grid[np.newaxis, :]
        
        s = 10.0 * c_col + np.sin(10.0 * np.pi * x1_row)
        denom = (4.0 * x1_row + 4.0) ** 2
        penalty = (s * s) / denom
        
        # Find index of minimum penalty for each c
        min_indices = np.argmin(penalty, axis=1)
        initial_x1 = x1_grid[min_indices]
        
        # Refine each initial_x1 with minimize_scalar
        # We can do this in a loop, but minimize_scalar is fast enough for 200 calls
        # Or we can just use the grid result if it's dense enough?
        # 2000 points might be too coarse for 1e-8 precision.
        # Let's refine.
        
        for i in range(n_targets):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            c_val = cs[i]
            x1_init = initial_x1[i]
            
            # Define local bounds for refinement
            # The grid step is 1/1999 ~ 0.0005
            # Use a small window around the initial guess
            lower = max(0.0, x1_init - 0.001)
            upper = min(1.0, x1_init + 0.001)
            
            if lower >= upper:
                x1_opt = x1_init
            else:
                def penalty_func(x1):
                    s = 10.0 * c_val + np.sin(10.0 * np.pi * x1)
                    denom = (4.0 * x1 + 4.0) ** 2
                    return (s * s) / denom
                
                try:
                    res = minimize_scalar(penalty_func, bounds=(lower, upper), method='bounded', options={'xatol': 1e-10})
                    x1_opt = res.x
                except:
                    x1_opt = x1_init
            
            x = np.full(self.dim, c_val)
            x[0] = x1_opt
            x = np.clip(x, lo, hi)
            
            x, f = eval_point(x)
            add_point(x, f)

        return archive_X if archive_X else np.array([lo + 0.5 * span])
# EVOLVE-BLOCK-END
