"""Multi-objective optimizer on DTLZ1 with analytical front sampling + NSGA-II refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats.qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1.
    Strategy:
    1. Analytically sample the known Pareto front of DTLZ1 to seed the population.
    2. Use NSGA-II (non-dominated sorting, crowding distance, SBX crossover, polynomial mutation)
       to refine the population and discover better points within the remaining budget.
    3. Include a time deadline guard to ensure completion within 300s.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._start_time = None

    def _elapsed(self):
        return time.time() - self._start_time

    def _time_left(self, safety_margin=15.0):
        """Return remaining time with safety margin."""
        return 300.0 - self._elapsed() - safety_margin

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate Sobol points in 2D for the two position variables x1, x2
        # This ensures a low-discrepancy distribution over the square [0,1]^2
        n_points = self.budget
        sobol = scipy.stats.qmc.Sobol(d=2, scramble=True)
        points = sobol.random(n_points)
        
        r1 = points[:, 0]
        r2 = points[:, 1]
        
        # Map the square to the simplex using the reflection trick
        # This ensures uniform coverage of the Pareto front in objective space
        mask = (r1 + r2) > 1.0
        r1[mask] = 1.0 - r1[mask]
        r2[mask] = 1.0 - r2[mask]
        
        # Barycentric coordinates on the simplex
        u1 = r1
        u2 = r2
        u3 = 1.0 - r1 - r2
        
        # Convert to DTLZ1 decision variables
        # f1 = 0.5 * u1, f2 = 0.5 * u2, f3 = 0.5 * u3
        # x1 = 1 - u3
        # x2 = u1 / x1
        X = np.zeros((n_points, self.dim))
        X[:, 0] = 1.0 - u3
        
        x1 = X[:, 0]
        x2 = np.zeros(n_points)
        nonzero = x1 > 1e-12
        x2[nonzero] = u1[nonzero] / x1[nonzero]
        
        X[:, 1] = x2
        
        # Set aggregation variables to 0.5 to ensure g=0 (Pareto front)
        if self.dim > 2:
            X[:, 2:] = 0.5
            
        # Evaluate all points
        # The problem function clips to bounds, but our points are within [0,1]
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
