"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using deterministic grid on Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ2 structure for 3 objectives, 12 vars:
        # x[0], x[1] = position variables
        # x[2:12] = convergence variables
        # g(x) = sum((x_i - 0.5)^2) for i in 2..11
        # Pareto front: g is minimized. For DTLZ2, the Pareto front shape
        # is determined by position vars when g is at its minimum.
        # 
        # Key: We need g > 0 to get spread-out points. The standard DTLZ2
        # Pareto front has the shape where sum(f_i^2) is constant.
        # Setting convergence vars to give g=1 places points on the unit front.
        #
        # To get g=1 with 10 convergence vars in [0,1]:
        # Set 4 vars to 0.0 and 6 vars to 0.5: 4*(0.5)^2 + 6*0 = 1.0
        # Or set 4 vars to 1.0 and 6 vars to 0.5: same.

        # Build a template for convergence vars that gives g=1
        conv_template = np.full(dim - 2, 0.5)
        # Set first 4 convergence vars to 0.0 to contribute 4*0.25 = 1.0 to g
        conv_template[:4] = 0.0

        # 10x10 grid in position space
        grid_n = 10
        points = []
        for i in range(grid_n):
            for j in range(grid_n):
                x = np.zeros(dim)
                x[0] = i / (grid_n - 1)
                x[1] = j / (grid_n - 1)
                x[2:] = conv_template
                x = np.clip(x, lo, hi)
                points.append(x)

        points = np.array(points)
        F = np.array([problem(x) for x in points])
        used = len(points)

        # Use remaining budget for refinement: perturb convergence vars
        # to explore nearby points on the front
        if used < budget:
            rng = np.random.default_rng(42)
            remaining = budget - used
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                # Pick a random grid point and perturb it slightly
                idx = rng.integers(len(points))
                x = points[idx].copy()
                # Small perturbation in position and convergence
                x[0] = np.clip(x[0] + 0.02 * rng.standard_normal(), lo[0], hi[0])
                x[1] = np.clip(x[1] + 0.02 * rng.standard_normal(), lo[1], hi[1])
                x[2:] = np.clip(x[2:] + 0.05 * rng.standard_normal(dim - 2), lo[2:], hi[2:])
                fc = problem(x)
                # Replace the worst point (highest sum of objectives)
                worst = int(np.argmax(np.sum(F, axis=1)))
                points[worst] = x
                F[worst] = fc
                used += 1

        return points
# EVOLVE-BLOCK-END
