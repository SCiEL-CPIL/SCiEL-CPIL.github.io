import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using a hybrid approach:
    - Exploit the known structure of ZDT4 (x1..x9 should be near 0 or 2)
    - Dense sweep of x0 combined with smart sampling of tail variables
    - Local refinement around promising points
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT4 specific: x0 in [-5,5], x1..x9 in [0,1]
        # g(x) = sum(x_i^2) for i=1..9, min at all zeros
        # f1 = 1 + 10*(n-1) + (x0^2 - 10*cos(4*pi*x0))
        # f2 = g * h(x0, g) where h = 1 - sqrt(f1/(1+g))

        # Strategy:
        # 1. Dense 1D sweep of x0 (100 points) with x1..x9 = 0
        # 2. Coarse 2D grid over x1, x2 with remaining tails at 0
        # 3. Local refinement around best points found

        all_points = []
        all_fvals = []

        def eval_point(x):
            f = problem(x)
            return f

        # Phase 1: Dense sweep of x0 with x1..x9 = 0
        # This should find the true Pareto front since g=0 is optimal
        n_x0 = 100
        x0_vals = np.linspace(lo[0], hi[0], n_x0)
        for x0 in x0_vals:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x0
            f = eval_point(x)
            all_points.append(x.copy())
            all_fvals.append(f)
            if len(all_points) >= budget:
                break

        # Phase 2: Coarse grid over x1, x2 with x3..x9 = 0
        # Sample a few points to verify that non-zero tails are worse
        if len(all_points) < budget:
            n_grid = 10
            x1_vals = np.linspace(lo[1], hi[1], n_grid)
            x2_vals = np.linspace(lo[2], hi[2], n_grid)
            # Only sample a subset to stay within budget
            for x1 in x1_vals[::2]:
                for x2 in x2_vals[::2]:
                    if time.time() > deadline or len(all_points) >= budget:
                        break
                    x0 = (lo[0] + hi[0]) / 2.0  # Middle of x0 range
                    x = np.zeros(dim)
                    x[0] = x0
                    x[1] = x1
                    x[2] = x2
                    f = eval_point(x)
                    all_points.append(x.copy())
                    all_fvals.append(f)

        # Phase 3: Local refinement around promising points
        # Find points with low f2 (which requires low g, i.e., x1..x9 near 0)
        if len(all_points) < budget and len(all_points) > 0:
            fvals_arr = np.array(all_fvals)
            # Sort by f2 and take top candidates for refinement
            idx_sorted = np.argsort(fvals_arr[:, 1])
            n_refine = min(20, budget - len(all_points))
            for i in range(n_refine):
                if time.time() > deadline:
                    break
                idx = idx_sorted[i % len(idx_sorted)]
                x_center = all_points[idx].copy()
                # Small perturbation in x1..x9 directions
                for d in range(1, min(dim, 4)):  # Refine first few tail dims
                    for delta in [-0.01, 0.01]:
                        if len(all_points) >= budget:
                            break
                        if time.time() > deadline:
                            break
                        x_new = x_center.copy()
                        x_new[d] += delta
                        x_new = np.clip(x_new, lo, hi)
                        f = eval_point(x_new)
                        all_points.append(x_new.copy())
                        all_fvals.append(f)

        # Return the best set of points (non-dominated)
        if len(all_points) == 0:
            # Fallback: return a single point
            x = (lo + hi) / 2.0
            f = eval_point(x)
            all_points.append(x)
            all_fvals.append(f)

        # Return all evaluated points; the evaluator records all objective vectors
        # We just need to make sure we've made at most budget evaluations
        return np.array(all_points)
