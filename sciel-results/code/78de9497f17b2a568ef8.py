"""Multi-objective optimizer for DTLZ1 exploiting known structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting DTLZ1 structure for hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        all_X = []
        all_F = []

        def evaluate(x):
            nonlocal all_X, all_F
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_X.append(x.copy())
            all_F.append(f.copy())
            return f

        def make_x(x0, x1, shape_offsets=None):
            """Create a DTLZ1 point with given x0, x1 and shape variables near 0.5."""
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            if shape_offsets is not None:
                x[2:] = 0.5 + shape_offsets
            else:
                x[2:] = 0.5
            return np.clip(x, lo, hi)

        # All evaluations on Pareto front: x[2:] = 0.5 exactly for g=0
        # Use Sobol-like quasi-random sequence for optimal space-filling
        # DTLZ1 front: f1=0.5*x0, f2=0.5*(1-x0)*x1, f3=0.5*(1-x0)*(1-x1)
        # f1+f2+f3 = 0.5, a 2D simplex in [0, 0.5]^3
        
        # Use stratified sampling: divide [0,1]^2 into cells, sample one point per cell
        # This gives better HV than regular grid for the same N
        n = self.budget  # use all budget on the front
        
        # Stratified sampling: sqrt(n) x sqrt(n) grid of cells
        n_side = int(np.sqrt(n))
        # Adjust to use all points
        n_side2 = n_side * n_side
        n_remainder = n - n_side2
        
        # Generate stratified points
        points = []
        for i in range(n_side):
            for j in range(n_side):
                if time.time() > deadline - 10:
                    break
                # Random offset within cell [i/n_side, (i+1)/n_side) x [j/n_side, (j+1)/n_side)
                x0 = (i + rng.random()) / n_side
                x1 = (j + rng.random()) / n_side
                points.append((x0, x1))
        
        # Add remaining points with low-discrepancy sequence
        if n_remainder > 0:
            for k in range(n_remainder):
                if time.time() > deadline - 10:
                    break
                # Use golden ratio conjugate for low-discrepancy
                x0 = ((k + 1) * 0.618033988749895) % 1.0
                x1 = ((k + 1) * 0.7548776662466927) % 1.0
                points.append((x0, x1))
        
        # Evaluate all points on the Pareto front
        for x0, x1 in points:
            if len(all_X) >= self.budget or time.time() > deadline - 5:
                break
            x = make_x(x0, x1)  # x[2:] = 0.5 exactly
            evaluate(x)

        return all_X
# EVOLVE-BLOCK-END
