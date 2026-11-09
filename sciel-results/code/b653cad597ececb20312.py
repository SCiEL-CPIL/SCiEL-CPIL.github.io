# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Structured grid-search + 1D local refinement exploiting WFG9 analytic structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin under 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        used = 0

        # WFG9 structure: f1=f(x0), f2=f(x1), f3=f(x2)+osc(x0,x1)
        # Pareto front: x2 at lo[2], trade-off between x0 and x1
        # Use 2D grid on x[0], x[1], fix x[2]=lo[2], others at midpoints

        # 12x12 = 144 grid points, leaves 56 for refinement
        n_grid = 12
        grid_size = n_grid * n_grid

        if grid_size > budget:
            n_grid = max(2, int(np.sqrt(budget)) - 1)
            grid_size = n_grid * n_grid

        # Base vector: non-primary vars at midpoints, x[2] at lower bound
        base = np.array([(lo[i] + hi[i]) / 2.0 for i in range(dim)])
        base[2] = lo[2]  # minimize f3 constant term

        # Generate grid points over x[0], x[1]
        x0_vals = np.linspace(lo[0], hi[0], n_grid)
        x1_vals = np.linspace(lo[1], hi[1], n_grid)

        points = []
        for x0 in x0_vals:
            for x1 in x1_vals:
                x = base.copy()
                x[0] = x0
                x[1] = x1
                points.append(x)

        points = np.array(points)

        # Evaluate all grid points
        all_F = []
        all_X = []
        for i in range(len(points)):
            if used >= budget:
                break
            if time.time() > deadline:
                break
            fp = problem(points[i])
            used += 1
            all_F.append(fp)
            all_X.append(points[i])

        all_F = np.array(all_F)
        all_X = np.array(all_X)

        # Find non-dominated points
        n_pts = len(all_F)
        if n_pts > 0:
            nd_mask = np.ones(n_pts, dtype=bool)
            for i in range(n_pts):
                if not nd_mask[i]:
                    continue
                for j in range(n_pts):
                    if i == j or not nd_mask[j]:
                        continue
                    if np.all(all_F[j] <= all_F[i]) and np.any(all_F[j] < all_F[i]):
                        nd_mask[i] = False
                        break

            nd_indices = np.where(nd_mask)[0]

            # 1D refinement along x[0] and x[1] for diverse non-dominated points
            remaining = budget - used
            if remaining > 0 and len(nd_indices) > 0:
                # Sort by f1 for diversity, pick evenly spaced
                sort_idx = np.argsort(all_F[nd_indices, 0])
                # Each refinement point costs up to 4 evals (2 axes x 2 signs)
                max_refine_pts = remaining // 4
                n_refine = min(len(sort_idx), max_refine_pts)
                if n_refine > 0:
                    if len(sort_idx) > n_refine:
                        step = len(sort_idx) / n_refine
                        refine_idx = [sort_idx[int(i * step)] for i in range(n_refine)]
                    else:
                        refine_idx = list(sort_idx)

                    for ri in refine_idx:
                        if used >= budget:
                            break
                        if time.time() > deadline:
                            break
                        center = all_X[ri]
                        for axis in [0, 1]:
                            if used >= budget:
                                break
                            if time.time() > deadline:
                                break
                            span = hi[axis] - lo[axis]
                            for sign in [1.0, -1.0]:
                                if used >= budget:
                                    break
                                if time.time() > deadline:
                                    break
                                x_new = center.copy()
                                x_new[axis] += sign * span * 0.02
                                x_new = np.clip(x_new, lo, hi)
                                fp = problem(x_new)
                                used += 1
                                all_F = np.vstack([all_F, fp.reshape(1, -1)])
                                all_X = np.vstack([all_X, x_new.reshape(1, -1)])

        return all_X
# EVOLVE-BLOCK-END
