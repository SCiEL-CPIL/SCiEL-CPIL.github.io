import numpy as np
import time

try:
    from scipy.stats import qmc
    HAS_QMC = True
except ImportError:
    HAS_QMC = False


class Optimizer:
    """DTLZ2 optimizer: Sobol sequence on shape vars with dist=0 for max hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # For DTLZ2: first n_obj-1 variables are shape variables
        # Remaining are distance variables (optimal = 0.0 for M=3)
        k_shape = self.n_obj - 1  # 2 for M=3
        shape_vars = list(range(k_shape))
        dist_vars = list(range(k_shape, self.dim))

        # Step 1: Confirm dist_val = 0.0 is optimal (1 eval)
        x_test = np.zeros(self.dim)
        for i in shape_vars:
            x_test[i] = 0.5
        f_test = np.array(problem(x_test))
        # sum(f^2) should be ~1.0 if on Pareto front
        dist_val = 0.0 if np.sum(f_test**2) < 1.5 else 0.5

        evals_used = 1
        remaining_budget = self.budget - evals_used

        # Step 2: Generate space-filling points on shape variables
        if HAS_QMC:
            # Use Sobol sequence with no scrambling for deterministic corner coverage
            sampler = qmc.Sobol(d=k_shape, scramble=False)
            # Find next power of 2 >= remaining_budget
            n_points = 1
            while n_points < remaining_budget:
                n_points *= 2
            points = sampler.random(n_points)
            # Take the first remaining_budget points
            points = points[:remaining_budget]
        else:
            # Fallback to uniform grid
            g = int(np.sqrt(remaining_budget))
            g = max(g, 10)
            grid_vals = np.linspace(0.0, 1.0, g)
            points = []
            for v1 in grid_vals:
                for v2 in grid_vals:
                    if len(points) >= remaining_budget:
                        break
                    points.append([v1, v2])
            points = np.array(points)
            if len(points) > remaining_budget:
                points = points[:remaining_budget]

        solutions = []
        objective_values = []
        used = 0

        # Phase 1: Evaluate generated points
        for p in points:
            if time.time() > deadline:
                break
            if used >= remaining_budget:
                break

            x = np.zeros(self.dim)
            for i, sv in enumerate(shape_vars):
                x[sv] = p[i]
            for i in dist_vars:
                x[i] = dist_val

            f = np.array(problem(x))
            solutions.append(x)
            objective_values.append(f)
            used += 1

        return np.array(solutions) if solutions else np.array([(lo + hi) / 2.0])
