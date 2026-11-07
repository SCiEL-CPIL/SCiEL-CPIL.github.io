import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 330s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use Sobol sequence for initial low-discrepancy spread
        sobol = qmc.Sobol(d=dim, scramble=True)
        n_init = min(budget, 64)
        # Get points in [0,1]^dim
        init_pts = sobol.random(n_init)
        X_init = lo + (hi - lo) * init_pts
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        if used >= budget:
            return X_init

        # Keep all initial points as archive
        archive_X = list(X_init)
        archive_F = list(F_init)

        # Main search loop: perturb points near the Pareto front
        remaining = budget - used
        # Focus mutations on non-dominated points
        for i in range(remaining):
            if time.time() > deadline:
                break

            # Find non-dominated points in current archive
            F_arr = np.array(archive_F)
            # Simple non-domination: point i is non-dominated if no other point dominates it
            nd_mask = np.ones(len(F_arr), dtype=bool)
            for k in range(len(F_arr)):
                if nd_mask[k]:
                    for j in range(len(F_arr)):
                        if j != k:
                            if np.all(F_arr[j] <= F_arr[k]) and np.any(F_arr[j] < F_arr[k]):
                                nd_mask[k] = False
                                break

            nd_idx = np.where(nd_mask)[0]
            if len(nd_idx) == 0:
                parent_idx = np.random.randint(len(archive_F))
            else:
                parent_idx = nd_idx[np.random.randint(len(nd_idx))]

            parent_x = archive_X[parent_idx]
            # Gaussian perturbation with adaptive step size
            step = 0.05 * (hi - lo)
            child = parent_x + step * np.random.randn(dim)
            child = np.clip(child, lo, hi)

            f_child = problem(child)
            used += 1

            # Add to archive
            archive_X.append(child)
            archive_F.append(f_child)

            # Prune dominated points to keep archive manageable
            if len(archive_X) > 500:
                F_all = np.array(archive_F)
                keep = np.ones(len(F_all), dtype=bool)
                for k in range(len(F_all)):
                    if keep[k]:
                        for j in range(len(F_all)):
                            if j != k and keep[j]:
                                if np.all(F_all[j] <= F_all[k]) and np.any(F_all[j] < F_all[k]):
                                    keep[k] = False
                                    break
                archive_X = [archive_X[k] for k in range(len(archive_X)) if keep[k]]
                archive_F = [archive_F[k] for k in range(len(archive_F)) if keep[k]]

        return archive_X[0] if archive_X else None
