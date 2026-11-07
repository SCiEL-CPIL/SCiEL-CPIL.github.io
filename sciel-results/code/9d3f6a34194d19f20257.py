"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid multi-objective optimizer: Tchebycheff decomposition + adaptive mutation + local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        archive_X = []
        archive_F = []
        used = 0

        def evaluate_and_add(x):
            nonlocal used
            if used >= budget:
                return
            if time.time() > deadline:
                return

            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            used += 1

            if len(archive_F) == 0:
                archive_X.append(x.copy())
                archive_F.append(f.copy())
                return

            F_arr = np.array(archive_F)

            # Check if new point is dominated
            mask_le = np.all(F_arr <= f, axis=1)
            mask_lt = np.any(F_arr < f, axis=1)
            if np.any(mask_le & mask_lt):
                return

            # Remove dominated points from archive
            mask_le2 = np.all(f <= F_arr, axis=1)
            mask_lt2 = np.any(f < F_arr, axis=1)
            to_remove = np.where(mask_le2 & mask_lt2)[0]

            for idx in sorted(to_remove, reverse=True):
                archive_X.pop(idx)
                archive_F.pop(idx)

            archive_X.append(x.copy())
            archive_F.append(f.copy())

        # Phase 1: Initialization with diverse sampling
        n_init = 50
        X_init = np.zeros((n_init, dim))
        n_rand = 25
        n_corner = 15
        n_mid = n_init - n_rand - n_corner

        X_init[:n_rand] = lo + span * np.random.random((n_rand, dim))

        for i in range(n_corner):
            X_init[n_rand + i] = np.where(np.random.random(dim) < 0.5, lo, hi)

        for i in range(n_mid):
            # Mid-range points with some randomness
            X_init[n_rand + n_corner + i] = lo + span * (0.2 + 0.6 * np.random.random(dim))

        for x in X_init:
            evaluate_and_add(x)

        # Phase 2: Global search using Tchebycheff-inspired weight-based exploration
        n_local_reserve = 100
        n_global = budget - used - n_local_reserve
        if n_global < 10:
            n_local_reserve = max(20, budget - used - 10)
            n_global = budget - used - n_local_reserve

        # Generate weight vectors for Tchebycheff decomposition
        n_weights = 20
        weights = np.linspace(0.05, 0.95, n_weights)
        weights = np.array([[w, 1.0 - w] for w in weights])

        # Track best solution per weight region
        best_per_weight = {}

        for i in range(n_global):
            if time.time() > deadline:
                break
            if used >= budget - n_local_reserve:
                break

            # Select weight vector (cycle through with some randomness)
            w_idx = i % n_weights
            w = weights[w_idx]

            # Select parent: use best known for this weight if available, else random archive
            if w_idx in best_per_weight and len(best_per_weight[w_idx]) > 0:
                parent = np.array(archive_X[best_per_weight[w_idx][0]])
            elif len(archive_X) > 0:
                idx = np.random.randint(len(archive_X))
                parent = np.array(archive_X[idx])
            else:
                parent = np.random.uniform(lo, hi)

            # Adaptive mutation with decreasing step size
            progress = used / budget
            base_scale = 0.4 * (1.0 - 0.7 * progress) + 0.03
            # Add some diversity with occasional large jumps
            if np.random.random() < 0.1:
                scale = base_scale * 2.0
            else:
                scale = base_scale

            child = parent + scale * span * np.random.standard_normal(dim)

            evaluate_and_add(child)

            # Update best per weight based on Tchebycheff scalarization
            if len(archive_F) > 0:
                F_arr = np.array(archive_F)
                # Tchebycheff scalarization: max_j w_j * (f_j - z_j)
                # Use current reference point from archive extremes
                z = np.array([np.min(F_arr[:, 0]), np.min(F_arr[:, 1])])
                scalarized = np.max(w * (F_arr - z), axis=1)
                best_idx = np.argmin(scalarized)

                if w_idx not in best_per_weight:
                    best_per_weight[w_idx] = []
                if len(best_per_weight[w_idx]) < 3:
                    best_per_weight[w_idx].append(best_idx)
                else:
                    # Replace worst in this weight group
                    group_F = F_arr[best_per_weight[w_idx]]
                    group_scalar = np.max(w * (group_F - z), axis=1)
                    worst_in_group = np.argmax(group_scalar)
                    best_per_weight[w_idx][worst_in_group] = best_idx

        # Phase 3: Local polish on diverse archive points
        if len(archive_X) > 0:
            F_arr = np.array(archive_F)

            # Select diverse points: use hypervolume contribution heuristic
            # Points that are on the convex hull of the objective space are valuable
            order_f1 = np.argsort(F_arr[:, 0])
            order_f2 = np.argsort(F_arr[:, 1])

            # Select up to 8 points: extremes and evenly spaced
            n_select = min(8, len(F_arr))
            if n_select <= 2:
                indices_to_optimize = list(range(len(F_arr)))
            else:
                # Take min f1, min f2, max f1, max f2, and evenly spaced in between
                idxs = set()
                idxs.add(order_f1[0])  # min f1
                idxs.add(order_f2[0])  # min f2
                idxs.add(order_f1[-1])  # max f1
                idxs.add(order_f2[-1])  # max f2

                # Add evenly spaced points
                for k in range(1, n_select - len(idxs) + 1):
                    pos = int(k * len(order_f1) / (n_select - len(idxs) + 1))
                    idxs.add(order_f1[pos])

                indices_to_optimize = list(idxs)[:n_select]

            for idx in indices_to_optimize:
                if time.time() > deadline:
                    break
                if used >= budget:
                    break

                x0 = np.array(archive_X[idx])
                f0 = np.array(archive_F[idx])

                # Tchebycheff-based local objective
                # Use weight based on position in objective space
                F_arr_full = np.array(archive_F)
                f1_min, f1_max = np.min(F_arr_full[:, 0]), np.max(F_arr_full[:, 0])
                f2_min, f2_max = np.min(F_arr_full[:, 1]), np.max(F_arr_full[:, 1])

                f1_norm = (f0[0] - f1_min) / (f1_max - f1_min + 1e-10)
                f2_norm = (f0[1] - f2_min) / (f2_max - f2_min + 1e-10)

                # Weight that emphasizes the "weak" dimension
                w = np.array([1.0 - f1_norm, 1.0 - f2_norm])
                w = np.maximum(w, 0.1)
                w = w / np.sum(w)

                def obj(x, w=w):
                    x_clipped = np.clip(x, lo, hi)
                    f = np.array(problem(x_clipped), dtype=np.float64)
                    return np.dot(w, f)

                remaining = budget - used
                maxfev = min(30, remaining)
                if maxfev < 2:
                    continue

                try:
                    res = minimize(obj, x0, method='Nelder-Mead',
                                   options={'maxiter': maxfev, 'maxfev': maxfev,
                                            'xatol': 1e-5, 'fatol': 1e-5,
                                            'adaptive': True})
                    x_best = np.clip(res.x, lo, hi)
                    evaluate_and_add(x_best)
                except Exception:
                    pass

                # Second polish with different weights for same point
                if used < budget and time.time() < deadline:
                    w2 = np.array([1.0, 1.0])
                    w2 = w2 / np.sum(w2)

                    def obj2(x, w=w2):
                        x_clipped = np.clip(x, lo, hi)
                        f = np.array(problem(x_clipped), dtype=np.float64)
                        return np.dot(w, f)

                    remaining = budget - used
                    maxfev2 = min(15, remaining)
                    if maxfev2 >= 2:
                        try:
                            res2 = minimize(obj2, x_best, method='Nelder-Mead',
                                           options={'maxiter': maxfev2, 'maxfev': maxfev2,
                                                    'xatol': 1e-6, 'fatol': 1e-6,
                                                    'adaptive': True})
                            x_best2 = np.clip(res2.x, lo, hi)
                            evaluate_and_add(x_best2)
                        except Exception:
                            pass

        if archive_X:
            return np.array(archive_X)
        return np.array([lo])
# EVOLVE-BLOCK-END
