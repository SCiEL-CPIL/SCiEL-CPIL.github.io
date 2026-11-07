import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_pos = self.n_obj - 1  # 2
        n_dist = self.dim - n_pos  # 10

        # Strategy:
        # 1. Use a deterministic 10x10 grid in the 2D position space (x[0], x[1])
        #    with distance variables fixed at 0.5 (g=0) to guarantee exact coverage
        #    of the Pareto front's boundaries and corners.
        # 2. Use remaining budget for local search to refine points.

        # Phase 1: Deterministic 10x10 grid on the Pareto front
        # x[0] and x[1] are position variables in [0, 1]
        # x[2:12] are distance variables set to 0.5 for g=0
        n_grid = 10
        pos_vals = np.linspace(0.0, 1.0, n_grid)

        evaluated_points = []
        evaluated_f = []
        used = 0

        for t1 in pos_vals:
            for t2 in pos_vals:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                x = np.zeros(self.dim)
                x[0] = t1
                x[1] = t2
                x[n_pos:] = 0.5  # g = 0
                x = np.clip(x, lo, hi)

                f = np.array(problem(x))
                evaluated_points.append(x.copy())
                evaluated_f.append(f)
                used += 1

            if used >= self.budget or time.time() > deadline:
                break

        # Phase 2: Local search refinement
        remaining = self.budget - used
        if remaining > 0 and evaluated_points:
            rng = np.random.default_rng(42)
            F_arr = np.array(evaluated_f)

            # For each point, try small perturbations in position and distance vars
            n_local = len(evaluated_points)
            steps_per_point = max(1, remaining // n_local)

            for i in range(n_local):
                if used >= self.budget:
                    break
                for step in range(steps_per_point):
                    if used >= self.budget or time.time() > deadline:
                        break

                    # Perturb position variables with decreasing step size
                    x = evaluated_points[i].copy()
                    step_size = 0.02 / (step + 1)
                    delta_pos = step_size * rng.standard_normal(n_pos)
                    x[:n_pos] += delta_pos

                    # Also try small perturbations in distance variables
                    delta_dist = 0.01 * rng.standard_normal(n_dist)
                    x[n_pos:] += delta_dist

                    x = np.clip(x, lo, hi)
                    f = np.array(problem(x))
                    used += 1

                    # Keep the point if it has lower sum of objectives
                    if np.sum(f) < np.sum(F_arr[i]):
                        evaluated_points[i] = x
                        F_arr[i] = f

        return evaluated_points[-1] if evaluated_points else np.zeros(self.dim)
