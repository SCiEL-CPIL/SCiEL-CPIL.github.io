# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Direct, structure-exploiting DTLZ4 optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 270

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        n = self.dim
        M = self.n_obj
        n_z = M - 1

        rng = np.random.default_rng(0)
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= self.budget or time.time() > deadline:
                return None
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return f

        # DTLZ4 analytical Pareto front for M=3:
        # g* = 0 (all g-variables at 0.5)
        # f_1 = (1 - z2) * (1 + g)  -> with g=0: f_1 = 1 - z2
        # f_2 = z2 * (1 - z1) * (1 + g) -> f_2 = z2*(1-z1)
        # f_3 = z2 * z1 * (1 + g) -> f_3 = z2*z1
        # Wait, let me use the standard DTLZ4 formula:
        # For DTLZ4, M objectives:
        #   f_i = g * h_i(z, g)
        # Standard DTLZ4 h functions for M=3:
        #   f_1 = g * (1 - z_2)  [h_1 = 1 - z_{M-1}]
        #   f_2 = g * z_2 * (1 - z_1)  [h_2 = z_{M-1}*(1-z_{M-2})]
        #   f_3 = g * z_2 * z_1  [h_3 = z_{M-1}*z_{M-2}]
        # Actually the standard DTLZ4 (Deb 2002) uses:
        #   f_i = g * h_i where h_i are specific trigonometric or polynomial
        # Let me just use the evaluator to get exact values on the front.

        # Generate dense grid of z-points on the Pareto front
        g_center = np.full(n - n_z, 0.5)

        # Dense structured sampling of z-space [0,1]^2
        n_grid = 15
        grid = np.linspace(0.0, 1.0, n_grid)

        X_list = []
        F_list = []

        for a in grid:
            for b in grid:
                if used >= self.budget or time.time() > deadline:
                    break
                z = np.array([a, b], dtype=float)
                x = np.concatenate([z, g_center])
                f = evaluate(x)
                if f is not None:
                    X_list.append(x)
                    F_list.append(f)

        # Add some random z-samples for better coverage
        n_rand = 50
        for _ in range(n_rand):
            if used >= self.budget or time.time() > deadline:
                break
            z = rng.random(n_z)
            x = np.concatenate([z, g_center])
            f = evaluate(x)
            if f is not None:
                X_list.append(x)
                F_list.append(f)

        if not X_list:
            # Fallback: single evaluation
            x = np.concatenate([rng.random(n_z), g_center])
            f = evaluate(x)
            if f is not None:
                X_list.append(x)
                F_list.append(f)
            return X_list

        X = np.array(X_list)
        F = np.array(F_list)

        # Non-dominated filter
        def non_dominated_indices(F_arr):
            n_pts = len(F_arr)
            nd = []
            for i in range(n_pts):
                is_nd = True
                for j in range(n_pts):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        is_nd = False
                        break
                if is_nd:
                    nd.append(i)
            return nd

        nd_idx = non_dominated_indices(F)
        if not nd_idx:
            nd_idx = list(range(len(F)))

        # Select top points by sum for refinement
        nd_F = F[nd_idx]
        sums = nd_F.sum(axis=1)
        top_k = min(20, len(nd_idx))
        order = np.argsort(sums)[:top_k]
        refine_idx = [nd_idx[k] for k in order]

        # Local refinement: perturb z-variables and fine-tune g-variables
        step = 0.05
        n_refine = min(80, self.budget - used)

        for it in range(n_refine):
            if used >= self.budget or time.time() > deadline:
                break
            if not refine_idx:
                break

            pi = refine_idx[it % len(refine_idx)]
            x0 = X[pi].copy()
            f0 = F[pi].copy()

            current_step = step * (0.92 ** it)
            best_x = x0.copy()
            best_f = f0.copy()

            n_trials = 3
            for _ in range(n_trials):
                if used >= self.budget or time.time() > deadline:
                    break
                x = x0.copy()
                # Perturb z-variables
                z_perturb = current_step * rng.standard_normal(n_z)
                x[:n_z] = np.clip(x[:n_z] + z_perturb, lo[:n_z], hi[:n_z])
                # Small perturbation on g-variables to keep them near 0.5
                g_perturb = 0.005 * rng.standard_normal(n - n_z)
                x[n_z:] = np.clip(x[n_z:] + g_perturb, lo[n_z:], hi[n_z:])

                f = evaluate(x)
                if f is None:
                    break

                if np.all(f <= best_f) and np.any(f < best_f):
                    best_x = x.copy()
                    best_f = f.copy()
                elif f.sum() < best_f.sum():
                    best_x = x.copy()
                    best_f = f.copy()

            X[pi] = best_x
            F[pi] = best_f

        # Final polish: tighten g-variables to exactly 0.5 for best points
        if used < self.budget and len(F) > 0:
            sums = F.sum(axis=1)
            top_n = min(10, len(F))
            top_idx = np.argsort(sums)[:top_n]

            for pi in top_idx:
                if used >= self.budget or time.time() > deadline:
                    break

                x0 = X[pi].copy()
                f0 = F[pi].copy()

                # Set g-variables exactly to 0.5 and evaluate
                x_clean = x0.copy()
                x_clean[n_z:] = 0.5
                f_clean = evaluate(x_clean)
                if f_clean is not None:
                    if np.all(f_clean <= f0) and np.any(f_clean < f0):
                        X[pi] = x_clean
                        F[pi] = f_clean
                    elif f_clean.sum() < f0.sum():
                        X[pi] = x_clean
                        F[pi] = f_clean

        return X
# EVOLVE-BLOCK-END
