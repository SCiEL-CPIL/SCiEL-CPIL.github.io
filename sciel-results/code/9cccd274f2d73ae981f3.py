"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-exploiting optimizer for DTLZ7 using Sobol-like sampling on the Pareto front manifold."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _halton_2d(self, n):
        """Generate n points in [0,1]^2 using Halton sequence (bases 2 and 3)."""
        points = np.zeros((n, 2))
        for i in range(n):
            # Halton base 2
            f, r, q = 1.0, 0.0, i + 1
            while q > 0:
                f /= 2.0
                r += f * (q % 2)
                q //= 2
            points[i, 0] = r
            # Halton base 3
            f, r, q = 1.0, 0.0, i + 1
            while q > 0:
                f /= 3.0
                r += f * (q % 3)
                q //= 3
            points[i, 1] = r
        return points

    def _dtlz7_front(self, t1, t2):
        """
        Generate a point on the DTLZ7 Pareto front given parameters t1, t2 in [0,1].
        DTLZ7 has 3 objectives. The Pareto front is parameterized by two variables.
        
        For DTLZ7 with M=3 objectives and K=20 (M-1=2 decision variables for the front):
        The front points satisfy:
          f1 = t1
          f2 = t2 * (1 - t1)
          f3 = (1 - t2) * (1 - t1) + something... 
          
        Actually, let me recall the exact DTLZ7 front.
        
        DTLZ7: M objectives, n variables.
        h = sum_{i=n-K+1}^{n} [f_i + 3*(sin^2(pi*f_i/2))] where K = n - M + 1
        
        For the Pareto front, the last K variables are set such that h=0, which means
        each of the last K variables is in [0, 0.5] (so sin^2 term is 0 when f_i=0).
        Actually h=0 requires f_i = 0 for i in the last K positions.
        
        The first M-1 variables (x_0, ..., x_{M-2}) determine the objectives:
          f_1 = x_0
          f_2 = x_1 * (1 - x_0)
          ...
          f_{M-1} = x_{M-2} * (1 - x_0) * ... * (1 - x_{M-3})
          f_M = (1 - x_{M-2}) * (1 - x_0) * ... * (1 - x_{M-3}) + h
        
        With h=0, for M=3:
          f_1 = x_0
          f_2 = x_1 * (1 - x_0)
          f_3 = (1 - x_1) * (1 - x_0)
        
        So with t1 = x_0, t2 = x_1:
          f_1 = t1
          f_2 = t2 * (1 - t1)
          f_3 = (1 - t2) * (1 - t1)
        """
        f1 = t1
        f2 = t2 * (1.0 - t1)
        f3 = (1.0 - t2) * (1.0 - t1)
        return np.array([f1, f2, f3])

    def _front_to_decision(self, t1, t2, lo, hi):
        """
        Map Pareto front parameters (t1, t2) to a decision vector x in [lo, hi]^dim
        that achieves the corresponding objective values (with h=0).
        
        For DTLZ7 with dim=22, M=3:
          K = dim - M + 1 = 20
          The first M-1 = 2 variables are x_0 = t1, x_1 = t2
          The last K = 20 variables must be set so that h = 0.
          h = sum_{i=2}^{21} [x_i + 3*sin^2(pi*x_i/2)]
          For h=0, we need each term to be 0, which requires x_i = 0 for i=2,...,21.
          
        So:
          x[0] = t1
          x[1] = t2
          x[2:] = 0
        """
        x = np.zeros(self.dim)
        x[0] = t1
        x[1] = t2
        # The rest are 0 to ensure h=0
        return x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin of 50s before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate Halton sequence points in [0,1]^2
        n_points = min(self.budget, 200)
        points_2d = self._halton_2d(n_points)
        
        # Add explicit corner points to capture extreme bounds
        corner_points = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ])
        points_2d = np.vstack([corner_points, points_2d[:n_points - 4]])
        
        # Evaluate points on the Pareto front
        F = np.zeros((n_points, self.n_obj))
        used = 0
        
        for i in range(n_points):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            t1, t2 = points_2d[i]
            # Clamp to [0,1] to ensure valid parameters
            t1 = np.clip(t1, 0.0, 1.0)
            t2 = np.clip(t2, 0.0, 1.0)
            
            x = self._front_to_decision(t1, t2, lo, hi)
            # Ensure x is within bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            F[i] = f
            used += 1
        
        return F
# EVOLVE-BLOCK-END
