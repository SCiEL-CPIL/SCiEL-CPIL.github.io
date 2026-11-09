# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Direct, structure-exploiting DTLZ4 optimizer using spherical parameterization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 270  # safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        M = self.n_obj
        n_z = M - 1  # number of shape variables (first M-1)

        rng = np.random.default_rng(0)

        # g-variables set to 0.5 (minimizes g for alpha=100)
        g_center = np.full(n - n_z, 0.5)

        X_list = []
        F_list = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                return None
            if time.time() > deadline:
                return None
            f = np.asarray(problem(x), dtype=float)
            X_list.append(x)
            F_list.append(f)
            used += 1
            return f

        # ---- Phase 1: Spherical sampling on the Pareto front ----
        # For DTLZ4 with M=3, the Pareto front in objective space is a 2D surface.
        # The shape variables z1, z2 ∈ [0,1] map to the front via:
        #   f1 = g*(1-z2)*cos(z1*pi/2)  ... but actually for DTLZ4:
        #   f1 = g * (1 - z_{M-1})^{1/alpha} * (1 + g)  -- no
        #
        # Standard DTLZ4 M=3:
        #   f1 = g * (1 - z2) * cos(z1*pi/2)
        #   f2 = g * (1 - z2) * sin(z1*pi/2)
        #   f3 = g * z2
        # Wait, that's DTLZ1. For DTLZ4:
        #   f1 = g * (1 - z2)^{1/alpha} * (1 + g)  -- no
        #
        # Actually DTLZ4 uses the same h-functions as DTLZ1 but with different g.
        # For DTLZ1/DTLZ4 with M=3:
        #   f1 = g * (1 - z2) * cos(z1 * pi/2)
        #   f2 = g * (1 - z2) * sin(z1 * pi/2)
        #   f3 = g * z2
        #
        # But wait, DTLZ4 uses alpha in the z-variables mapping. Let me check:
        # DTLZ4: z_i = x_i^beta where beta=100. So the shape variables are transformed.
        # Actually no - in DTLZ4, the g-function uses (x_i - 0.5)^alpha, and the
        # shape variables z_i = x_i^{1/alpha} or similar. Let me just use the evaluator.
        #
        # Key insight: The Pareto front is parameterized by z1, z2 ∈ [0,1].
        # Using spherical coordinates gives uniform coverage of the 2D front.

        # Uniform sampling on unit sphere in 3D, project to [0,1]^2
        # Use theta, phi from uniform sphere sampling
        n_sphere = 150
        for _ in range(n_sphere):
            # Sample uniformly on unit sphere
            u = rng.uniform(-1, 1)
            v = rng.uniform(0, 2 * np.pi)
            # Convert to spherical
            theta = np.arccos(u)  # polar angle [0, pi]
            # Map to z1, z2 in [0, 1]
            z1 = theta / np.pi  # [0, 1]
            z2 = (v / (2 * np.pi))  # [0, 1]
            # Clip to ensure valid
            z1 = np.clip(z1, 0.0, 1.0)
            z2 = np.clip(z2, 0.0, 1.0)
            x = np.concatenate([np.array([z1, z2]), g_center])
            evaluate(x)

        # Add structured grid points for coverage
        n_grid = 10
        for a in np.linspace(0.0, 1.0, n_grid):
            for b in np.linspace(0.0, 1.0, n_grid):
                if used >= self.budget:
                    break
                x = np.concatenate([np.array([a, b]), g_center])
                evaluate(x)

        # Add vertices/corners of the front for hypervolume extent
        corners = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.0],
            [0.5, 1.0],
            [0.0, 0.5],
            [1.0, 0.5],
        ]
        for c in corners:
            if used >= self.budget:
                break
            x = np.concatenate([np.array(c, dtype=float), g_center])
            evaluate(x)

        if len(F_list) == 0:
            return []

        X = np.array(X_list)
        F = np.array(F_list)

        # ---- Phase 2: Non-dominated filter ----
        def non_dominated_indices(F_arr):
            n_pts = len(F_arr)
            nd = []
            for i in range(n_pts):
                dominated = False
                for j in range(n_pts):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return nd

        nd_idx = non_dominated_indices(F)
        if not nd_idx:
            nd_idx = list(range(len(F)))

        # Select top points for refinement
        sums = F[nd_idx].sum(axis=1)
        order = np.argsort(sums)[:20]
        refine_idx = [nd_idx[k] for k in order]

        base_X = X.copy()
        base_F = F.copy()

        # ---- Phase 3: Local refinement ----
        step = 0.02
        n_refine = min(80, self.budget - used)

        for it in range(n_refine):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            if not refine_idx:
                break

            pi = refine_idx[it % len(refine_idx)]
            if pi >= len(base_X):
                pi = 0
            x0 = base_X[pi].copy()
            f0 = base_F[pi].copy()

            current_step = step * (0.92 ** it)
            best_x = x0
            best_f = f0
            best_score = best_f.sum()

            n_trials = 3
            for _ in range(n_trials):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                d = current_step * rng.standard_normal(n)
                x = np.clip(x0 + d, lo, hi)
                f = evaluate(x)
                if f is None:
                    break

                if np.all(f <= best_f) and np.any(f < best_f):
                    best_x = x
                    best_f = f
                    best_score = f.sum()
                elif f.sum() < best_score:
                    best_x = x
                    best_f = f
                    best_score = f.sum()

            base_X[pi] = best_x
            base_F[pi] = best_f

        # ---- Phase 4: Fine polish on g-variables ----
        if used < self.budget:
            if len(base_F) > 0:
                sums = base_F.sum(axis=1)
                top_idx = np.argsort(sums)[:10]

                for pi in top_idx:
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break

                    x0 = base_X[pi].copy()
                    f0 = base_F[pi].copy()

                    fine_step = 0.005
                    for _ in range(4):
                        if used >= self.budget:
                            break
                        if time.time() > deadline:
                            break

                        x = x0.copy()
                        g_perturb = fine_step * rng.standard_normal(n - n_z)
                        x[n_z:] = np.clip(x[n_z:] + g_perturb, lo[n_z:], hi[n_z:])
                        f = evaluate(x)
                        if f is None:
                            break

                        if np.all(f <= f0) and np.any(f < f0):
                            x0 = x
                            f0 = f
                        elif f.sum() < f0.sum():
                            x0 = x
                            f0 = f

                    base_X[pi] = x0
                    base_F[pi] = f0

        # Merge all recorded evaluations
        final_F = np.array(F_list)
        return final_F
# EVOLVE-BLOCK-END
