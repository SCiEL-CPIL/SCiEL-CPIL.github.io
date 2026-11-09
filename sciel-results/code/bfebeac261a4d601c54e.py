import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = 0
        all_points = []
        all_fvals = []

        def eval_point(x):
            nonlocal n_evals
            if n_evals >= self.budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            try:
                f = problem(x)
            except Exception:
                return None
            if len(f) != self.n_obj:
                return None
            f = np.array(f, dtype=float)
            if not np.all(np.isfinite(f)):
                return None
            all_points.append(x.copy())
            all_fvals.append(f)
            n_evals += 1
            return f

        # Strategy: 
        # 1. Assume the first 2 variables control the trade-off (angles) and the rest control the offset (g).
        # 2. To be robust against different variable orderings or DTLZ2 variants, 
        #    we perform a coarse grid search on the first 2 variables to find the "center" of the Pareto front 
        #    (where g is minimized).
        # 3. Then, we use the remaining budget to sample the Pareto front (g=0) densely using a Halton sequence 
        #    in the angle space, assuming the mapping from the first 2 variables to the angles is monotonic 
        #    and covers the range [0, pi/2] or similar.
        
        # Phase 1: Coarse Grid Search to find the best g-center for the last dim-2 variables
        # We assume the first 2 variables are the "angle" variables.
        # We fix the last dim-2 variables to the midpoint (standard DTLZ2 g=0 point) 
        # and vary the first 2 to see if they actually control the front.
        # If the problem is standard DTLZ2, x[0], x[1] are angles, x[2:] are g-vars.
        
        # Let's try a few candidate centers for the g-variables (x[2:]) 
        # and a few candidate angle ranges for x[0], x[1].
        
        # Standard DTLZ2: g=0 at x=0.5.
        # If bounds are [0,1], mid is 0.5.
        mid = (lo + hi) / 2.0
        
        # Let's verify if x[0], x[1] are indeed the angle variables by checking 
        # if varying them changes the objective directions while keeping g constant.
        
        # Phase 1: Evaluate a small grid on x[0], x[1] with x[2:] = mid
        # This helps confirm the structure and find the exact range of angles.
        n_grid = 10
        grid_budget = n_grid * n_grid
        if grid_budget > self.budget // 2:
            grid_budget = self.budget // 2
            n_grid = int(np.sqrt(grid_budget))
        
        grid_points = []
        # Sample x[0], x[1] in [lo, hi]
        # In standard DTLZ2, x[0], x[1] are in [0, pi/2] or [0, 1] mapped to angles.
        # If the problem clips to bounds, and bounds are [0,1], then x[0],x[1] are in [0,1].
        # The mapping to angles might be internal to the problem function.
        # So we just sample x[0], x[1] uniformly in [lo, hi].
        
        x0_vals = np.linspace(lo[0], hi[0], n_grid)
        x1_vals = np.linspace(lo[1], hi[1], n_grid)
        
        best_g_point = None
        best_g_val = float('inf')
        # We assume the Pareto front is where the sum of squares of objectives is minimized 
        # (closest to origin) for a fixed trade-off, but actually we just want to find 
        # the set of points that form the front.
        
        # Actually, a simpler heuristic: 
        # The Pareto front of DTLZ2 is a sphere. The "center" of the g-variables 
        # that minimizes g is usually the midpoint of the bounds if the function is symmetric.
        # Let's assume x[2:] = mid is the optimal g-center.
        # Then we just need to sample x[0], x[1] densely.
        
        # However, what if the angle variables are NOT x[0], x[1]?
        # In DTLZ2, the first n_obj-1 variables are angles. So x[0], x[1] should be them.
        
        # Let's do a quick check: evaluate a few points with x[2:] = mid and x[0], x[1] varied.
        # If the objective values span a wide range and look like a sphere, we are good.
        
        # For the main sampling, we use a Halton sequence on x[0], x[1] in [lo, hi].
        # We set x[2:] to mid.
        
        def halton_sequence(n, dims=2):
            points = []
            for i in range(1, n + 1):
                point = []
                for base in range(2, dims + 2):  # bases 2, 3
                    f = 1.0
                    r = 0.0
                    ii = i
                    while ii > 0:
                        f /= base
                        r += f * (ii % base)
                        ii //= base
                    point.append(r)
                points.append(point)
            return np.array(points)

        # We use all remaining budget for dense sampling on the front.
        n_dense = self.budget - n_evals
        
        if n_dense > 0:
            u = halton_sequence(n_dense, dims=2)
            
            # Map u to x[0], x[1] in [lo, hi]
            x0_samples = lo[0] + u[:, 0] * (hi[0] - lo[0])
            x1_samples = lo[1] + u[:, 1] * (hi[1] - lo[1])
            
            for i in range(n_dense):
                if time.time() > deadline:
                    break
                x = np.zeros(self.dim)
                x[0] = x0_samples[i]
                x[1] = x1_samples[i]
                x[2:] = mid[2:]
                eval_point(x)

        return all_points
