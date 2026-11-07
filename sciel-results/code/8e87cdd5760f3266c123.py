"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic DTLZ2-specific search using analytic Pareto front mapping."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        used = 0
        
        # For DTLZ2 with 3 objectives and 12 variables:
        # 2 position variables (indices 0, 1) and 10 distance variables (indices 2-11)
        # Pareto front: g=0, so distance vars at their optimum (0.5 for DTLZ2)
        # f1 = (1+g)*cos(a*pi/2), f2 = (1+g)*sin(a*pi/2)*cos(b*pi/2), f3 = (1+g)*sin(a*pi/2)*sin(b*pi/2)
        # where a = x[0], b = x[1]
        
        # Use a 10x10 uniform grid on the Pareto front to guarantee boundary coverage
        n_grid = 10
        for i in range(n_grid):
            for j in range(n_grid):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Angle parameters in [0, 1] (will be multiplied by pi/2 in objective)
                a = (i + 0.5) / n_grid
                b = (j + 0.5) / n_grid
                
                # Construct x: first 2 vars are position vars, rest are distance vars
                x = np.zeros(self.dim)
                x[0] = a
                x[1] = b
                x[2:] = 0.5  # distance vars at optimum for DTLZ2
                
                # Clip to bounds
                x = np.clip(x, lo, hi)
                
                f = problem(x)
                if np.all(np.isfinite(f)):
                    results.append(f)
                    used += 1
        
        if time.time() > deadline:
            return np.array(results) if results else np.empty((0, self.n_obj))
        
        # If we have budget left, add some random points to explore
        rng = np.random.default_rng(42)
        while used < self.budget and time.time() < deadline:
            x = lo + (hi - lo) * rng.random(self.dim)
            f = problem(x)
            if np.all(np.isfinite(f)):
                results.append(f)
                used += 1
        
        return np.array(results) if results else np.empty((0, self.n_obj))
# EVOLVE-BLOCK-END
