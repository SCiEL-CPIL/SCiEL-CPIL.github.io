"""Multi-objective optimizer on DTLZ7 using structured grid + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structured grid search on DTLZ7's Pareto-relevant variables with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ7: 22 variables. x[0], x[1] are Pareto-relevant (when g=0).
        # x[2:] should be 0 to minimize g.
        # f1 = x[0]
        # f2 = (1-x[0]) * h(x[0], x[1], g)
        # f3 = 1 + g
        # h(x0, x1, g) = 1 + g/x1 * (x1*sin(10*pi*x1 + pi) + (x1 - x0^2 - 1) + 2)
        # When g=0 and x1 > 0: h simplifies, but we just use the problem function.

        # Phase 1: Structured grid on x[0] and x[1]
        # Use a grid that covers [0,1] for both variables
        # Budget allocation: ~121 for grid (11x11), rest for refinement

        grid_size = 11
        grid_x0 = np.linspace(0, 1, grid_size)
        grid_x1 = np.linspace(0, 1, grid_size)

        evaluated = []
        eval_count = 0

        def eval_point(x):
            nonlocal eval_count
            x = np.asarray(x, dtype=float)
            # Clip to bounds
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            eval_count += 1
            return x, f

        # Phase 1: Grid evaluation
        for x0 in grid_x0:
            for x1 in grid_x1:
                if eval_count >= self.budget:
                    break
                if time.time() > deadline:
                    break
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                # x[2:] = 0 to minimize g
                x, f = eval_point(x)
                evaluated.append((x, f))
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

        # Get all evaluated points
        if not evaluated:
            # Fallback: random points
            rng = np.random.default_rng(0)
            for _ in range(min(20, self.budget)):
                x = lo + (hi - lo) * rng.random(self.dim)
                x, f = eval_point(x)
                evaluated.append((x, f))
            return np.array([x for x, f in evaluated])

        F_all = np.array([f for x, f in evaluated])
        X_all = np.array([x for x, f in evaluated])

        # Phase 2: Local refinement around promising (non-dominated) points
        # Find non-dominated points
        n_pts = len(F_all)
        dominated = np.zeros(n_pts, dtype=bool)
        for i in range(n_pts):
            for j in range(n_pts):
                if i != j:
                    if np.all(F_all[j] <= F_all[i]) and np.any(F_all[j] < F_all[i]):
                        dominated[i] = True
                        break

        nondom_idx = np.where(~dominated)[0]

        if len(nondom_idx) == 0:
            nondom_idx = np.arange(n_pts)

        # Refine around non-dominated points
        rng = np.random.default_rng(1)
        n_refine = min(self.budget - eval_count, 80)

        for _ in range(n_refine):
            if eval_count >= self.budget:
                break
            if time.time() > deadline:
                break

            # Pick a random non-dominated point to refine
            idx = nondom_idx[rng.integers(len(nondom_idx))]
            center = X_all[idx].copy()

            # Small perturbation in x[0] and x[1]
            step = 0.02
            dx0 = rng.uniform(-step, step)
            dx1 = rng.uniform(-step, step)

            new_x0 = np.clip(center[0] + dx0, 0, 1)
            new_x1 = np.clip(center[1] + dx1, 0, 1)

            x = center.copy()
            x[0] = new_x0
            x[1] = new_x1
            # Keep x[2:] = 0

            x, f = eval_point(x)
            evaluated.append((x, f))

        # Phase 3: If budget remains, do a finer grid in high-value regions
        if eval_count < self.budget and time.time() < deadline:
            # Focus on regions where f2 is small (left side of front) and f3 is small
            # Do a denser grid in [0, 0.5] x [0, 0.5]
            fine_size = 5
            fine_x0 = np.linspace(0, 0.5, fine_size)
            fine_x1 = np.linspace(0, 0.5, fine_size)

            for x0 in fine_x0:
                for x1 in fine_x1:
                    if eval_count >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    x = np.zeros(self.dim)
                    x[0] = x0
                    x[1] = x1
                    x, f = eval_point(x)
                    evaluated.append((x, f))
                if time.time() > deadline:
                    break

        # Return all evaluated X points (the return value is ignored by evaluator,
        # but we return something valid)
        return np.array([x for x, f in evaluated])
# EVOLVE-BLOCK-END
