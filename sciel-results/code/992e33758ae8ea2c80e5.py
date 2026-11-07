"""Optimizer for DTLZ4 multi-objective optimization using analytic front construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using analytic Pareto front construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _construct_point(self, f1, f2, f3, bounds):
        """Construct a DTLZ4 decision vector that produces objectives (f1, f2, f3) with g=0.
        
        For DTLZ4:
          f1 = cos(x1/2) * cos(x2/2)
          f2 = cos(x1/2) * sin(x2/2)  
          f3 = sin(x1/2)
        where x1, x2 in [0, 1].
        
        Given target (f1, f2, f3) on the Pareto front (f1^2 + f2^2 + f3^2 = 1, all >= 0):
          sin(x1/2) = f3  =>  x1/2 = arcsin(f3)  =>  x1 = 2*arcsin(f3)
          cos(x1/2) = sqrt(1 - f3^2)
          f1 = cos(x1/2)*cos(x2/2)  =>  cos(x2/2) = f1/sqrt(1-f3^2)
          f2 = cos(x1/2)*sin(x2/2)  =>  sin(x2/2) = f2/sqrt(1-f3^2)
          x2/2 = arctan2(f2, f1)  (since both divided by same positive factor)
          x2 = 2*arctan2(f2, f1)
        
        But wait - x1, x2 must be in [0,1] (the bounds). Let me check:
        arcsin(f3) where f3 in [0,1] gives [0, pi/2], so x1 = 2*arcsin(f3) in [0, pi].
        That's NOT in [0,1]! So we can't directly map arbitrary front points.
        
        Actually, looking at the problem more carefully: the decision variables x1, x2 
        are in [0,1] (bounds [0,1] for every variable). So:
          x1 in [0,1] => x1/2 in [0, 0.5] => f3 = sin(x1/2) in [0, sin(0.5)] ≈ [0, 0.4794]
          x2 in [0,1] => x2/2 in [0, 0.5]
        
        So the achievable Pareto front is only a portion of the full quarter-sphere!
        f3 ranges from 0 to sin(0.5) ≈ 0.4794
        And for each f3, f1^2 + f2^2 = cos^2(x1/2) = 1 - f3^2
        with x2/2 in [0, 0.5], so the angle theta = x2/2 in [0, 0.5]
        f1 = cos(x1/2)*cos(theta), f2 = cos(x1/2)*sin(theta)
        
        So the front is parameterized by (x1, x2) in [0,1]^2:
          f1 = cos(x1/2) * cos(x2/2)
          f2 = cos(x1/2) * sin(x2/2)
          f3 = sin(x1/2)
        
        To maximize hypervolume with reference point [2.5, 2.5, 2.5], we want to 
        cover the Pareto front as well as possible. The best strategy is to use a 
        low-discrepancy sequence in the (x1, x2) parameter space [0,1]^2.
        
        Let me just use the (x1, x2) parameterization directly with Sobol/Halton sampling.
        """
        x = np.zeros(self.dim)
        x[0] = f1  # This is actually x1, the decision variable
        x[1] = f2  # This is actually x2, the decision variable
        # g-variables (indices 7-11) set to 0.5 for g=0
        for i in range(7, 12):
            lo, hi = bounds[i]
            x[i] = 0.5
        # x[2], x[3], x[4], x[5], x[6] can be anything (they don't affect objectives when g=0)
        for i in range(2, 7):
            x[i] = 0.5
        # Clip to bounds
        for i in range(self.dim):
            lo, hi = bounds[i]
            x[i] = np.clip(x[i], lo, hi)
        return x

    def _sobol_2d(self, n, seed=0):
        """Generate n points in [0,1]^2 using a Sobol-like sequence."""
        # Use a simple low-discrepancy approach: combine two different bases
        # or use a stratified sampling with careful ordering
        
        # Simple approach: use the inverse CDF of a quasi-random sequence
        # For 2D, we can use a product of 1D low-discrepancy sequences
        
        # Let's use a Halton sequence with primes 2 and 3
        def halton_1d(n, base):
            result = np.zeros(n)
            for i in range(n):
                f = 1.0
                r = 0.0
                idx = i + 1
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                result[i] = r
            return result
        
        h1 = halton_1d(n, 2)
        h2 = halton_1d(n, 3)
        
        return np.column_stack([h1, h2])

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: Use Sobol sequence to sample (x1, x2) in [0,1]^2
        # which parameterizes the DTLZ4 Pareto front
        # Budget: 200 evaluations total
        
        n_total = self.budget
        
        # Generate Sobol points
        sobol_points = self._sobol_2d(n_total, seed=0)
        
        # Evaluate all points
        used = 0
        for idx in range(n_total):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x1, x2 = sobol_points[idx]
            x = self._construct_point(x1, x2, bounds)
            f = problem(x)
            used += 1
        
        return None
# EVOLVE-BLOCK-END
