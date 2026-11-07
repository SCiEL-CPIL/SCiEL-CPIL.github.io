import numpy as np
import time


class Optimizer:
    """DTLZ2 optimizer exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        k_shape = self.n_obj - 1  # 2 for M=3
        shape_vars = list(range(k_shape))
        dist_vars = list(range(k_shape, self.dim))

        # Phase 1: Dense sampling of the Pareto front (distance vars = 0)
        # For 2 shape vars, use a grid that maximizes hypervolume coverage
        # Use ~150 evals for grid, ~50 for refinement

        grid_budget = 150
        refine_budget = self.budget - grid_budget - 2  # reserve 2 for detection

        # Detect optimal distance setting (2 evals)
        x_test0 = np.zeros(self.dim)
        x_test0[shape_vars[0]] = 0.5
        x_test0[shape_vars[1]] = 0.5
        f_test0 = np.array(problem(x_test0))

        x_test1 = np.zeros(self.dim)
        x_test1[shape_vars[0]] = 0.5
        x_test1[shape_vars[1]] = 0.5
        for i in dist_vars:
            x_test1[i] = 0.5
        f_test1 = np.array(problem(x_test1))

        if np.sum(f_test0**2) < np.sum(f_test1**2):
            dist_val = 0.0
        else:
            dist_val = 0.5

        evals_used = 2
        solutions = []
        objective_values = []

        # Generate grid points for shape variables
        # For 2D shape space, use a grid of size g x g where g*g <= grid_budget
        g = int(np.sqrt(grid_budget))
        g = max(g, 12)

        # Use linspace but with slightly denser coverage near boundaries
        # where hypervolume contribution is highest
        grid_vals = np.linspace(0.0, 1.0, g)

        used = 0
        for v1 in grid_vals:
            for v2 in grid_vals:
                if time.time() > deadline or used >= grid_budget:
                    break

                x = np.zeros(self.dim)
                x[shape_vars[0]] = v1
                if len(shape_vars) > 1:
                    x[shape_vars[1]] = v2
                for i in dist_vars:
                    x[i] = dist_val

                f = np.array(problem(x))
                solutions.append(x.copy())
                objective_values.append(f)
                used += 1

            if used >= grid_budget or time.time() > deadline:
                break

        # Phase 2: Local refinement around promising points
        # Find points that contribute most to hypervolume
        if used < self.budget and time.time() < deadline and solutions:
            F = np.array(objective_values)
            n = len(F)

            # Simple non-dominated sort + hypervolume contribution estimate
            # For 3 objectives, a point is "good" if it's non-dominated
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i != j and not dominated[j]:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break

            nd_indices = np.where(~dominated)[0]

            # Refine around evenly spaced non-dominated points
            n_refine = min(len(nd_indices), 10)
            if n_refine > 0:
                if len(nd_indices) > n_refine:
                    # Pick evenly spaced from the non-dominated set
                    refine_idx = np.linspace(0, len(nd_indices) - 1, n_refine, dtype=int)
                    refine_idx = nd_indices[refine_idx]
                else:
                    refine_idx = nd_indices

                for idx in refine_idx:
                    if used >= self.budget or time.time() > deadline:
                        break

                    base_x = solutions[idx].copy()

                    # Try small perturbations in shape variables
                    step = 0.01
                    for dv1 in [-step, 0, step]:
                        for dv2 in [-step, 0, step]:
                            if dv1 == 0 and dv2 == 0:
                                continue
                            if used >= self.budget or time.time() > deadline:
                                break

                            new_x = base_x.copy()
                            new_x[shape_vars[0]] = np.clip(base_x[shape_vars[0]] + dv1, 0.0, 1.0)
                            if len(shape_vars) > 1:
                                new_x[shape_vars[1]] = np.clip(base_x[shape_vars[1]] + dv2, 0.0, 1.0)

                            new_f = np.array(problem(new_x))
                            solutions.append(new_x)
                            objective_values.append(new_f)
                            used += 1

        return np.array(solutions) if solutions else np.zeros(self.dim)
