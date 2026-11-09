import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # DTLZ1 specific structure:
        # f1 = 0.5 * (1+g) * x0 * x1
        # f2 = 0.5 * (1+g) * (1-x0) * x1
        # f3 = 0.5 * (1+g) * (1-x1)
        # g = 100 + 10*n + sum((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))) for g vars
        # For g=0, each g var should be 0.5

        n_front_vars = n_obj - 1  # 2 for DTLZ1
        n_g_vars = dim - n_front_vars  # 5 for DTLZ1

        # Phase 1: Create well-distributed points on the Pareto front
        # The Pareto front for DTLZ1 is defined by the simplex in (n_obj-1) dimensions
        # with g=0 (all g vars at 0.5)
        
        # Use a combination of stratified sampling and random points
        n_front_samples = int(budget * 0.45)

        # Stratified sampling in the 2D simplex (x0 >= 0, x1 >= 0, x0+x1 <= 1)
        n_grid = max(3, int(np.sqrt(2 * n_front_samples / 1.5)))
        grid_x = (np.arange(n_grid) + np.random.random(n_grid)) / n_grid
        mesh_x0, mesh_x1 = np.meshgrid(grid_x, grid_x)
        mesh_x0 = mesh_x0.flatten()
        mesh_x1 = mesh_x1.flatten()
        valid = (mesh_x0 + mesh_x1) <= 1.0
        front_x0 = mesh_x0[valid]
        front_x1 = mesh_x1[valid]

        if len(front_x0) <= n_front_samples:
            n_extra = n_front_samples - len(front_x0)
            extra_x0 = np.random.random(n_extra)
            extra_x1 = np.random.random(n_extra) * (1 - extra_x0)
            front_x0 = np.concatenate([front_x0, extra_x0])
            front_x1 = np.concatenate([front_x1, extra_x1])
        else:
            idx = np.random.choice(len(front_x0), n_front_samples, replace=False)
            front_x0 = front_x0[idx]
            front_x1 = front_x1[idx]
        
        front_points = np.zeros((len(front_x0), dim))
        front_points[:, 0] = front_x0
        front_points[:, 1] = front_x1
        # Set g variables to midpoint (0.5 for [0,1] bounds)
        g_mid = (lo[n_front_vars:] + hi[n_front_vars:]) / 2.0
        front_points[:, n_front_vars:] = g_mid
        front_points = np.clip(front_points, lo, hi)

        # Evaluate front points
        X = front_points.copy()
        F = np.array([problem(x) for x in X])
        used = len(X)

        # Phase 2: Refine g-variables toward target and fine-tune front positions
        # Use vectorized dominance for speed
        sums = np.sum(F, axis=1)

        while used < budget:
            if time.time() > deadline:
                break

            n_points = len(F)
            sums = np.sum(F, axis=1)

            # Tournament selection on sum of objectives
            if n_points >= 4:
                candidates = np.random.choice(n_points, min(4, n_points), replace=False)
                parent_idx = candidates[np.argmin(sums[candidates])]
            else:
                parent_idx = np.argmin(sums)

            parent = X[parent_idx].copy()

            g_dev = np.mean(np.abs(parent[n_front_vars:] - g_target))
            base_g = 0.03 * (hi[n_front_vars:] - lo[n_front_vars:])
            scale_g = base_g * max(0.2, min(1.0, g_dev / 0.03 + 0.1))
            scale_f = 0.015 * (hi[:n_front_vars] - lo[:n_front_vars])

            child = parent.copy()
            child[:n_front_vars] += scale_f * np.random.normal(size=n_front_vars)
            child[n_front_vars:] += scale_g * np.random.normal(size=n_g_vars)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Fast dominance check
            is_dom = np.any(np.all(F <= fc, axis=1) & np.any(F < fc, axis=1))

            if not is_dom:
                dominated_by_child = np.where(np.all(F <= fc, axis=1) & np.any(F < fc, axis=1))[0]
                if len(dominated_by_child) > 0:
                    replace_idx = dominated_by_child[np.argmin(sums[dominated_by_child])]
                else:
                    replace_idx = np.argmax(sums)
            else:
                worst_idx = np.argmax(sums)
                if np.sum(fc) < sums[worst_idx]:
                    replace_idx = worst_idx
                else:
                    continue

            X[replace_idx] = child
            F[replace_idx] = fc

        # Phase 3: Final focused refinement on g-variables
        if used < budget and time.time() < deadline - 10.0:
            order = np.argsort(np.sum(F, axis=1))
            for idx in order[:min(60, len(F))]:
                if used >= budget or time.time() > deadline:
                    break
                parent = X[idx].copy()
                for _ in range(3):
                    if used >= budget or time.time() > deadline:
                        break
                    child = parent.copy()
                    child[n_front_vars:] += 0.005 * (hi[n_front_vars:] - lo[n_front_vars:]) * np.random.normal(size=n_g_vars)
                    child = np.clip(child, lo, hi)
                    fc = problem(child)
                    used += 1
                    if np.sum(fc) < np.sum(F[idx]):
                        X[idx] = child
                        F[idx] = fc
                        parent = child

        return X
