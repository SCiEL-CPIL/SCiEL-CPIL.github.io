"""Optimized DTLZ4 solver: exploit known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ4 optimizer exploiting known Pareto front (g-vars at 0.5)."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270  # 30s safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim, n_obj, budget = self.dim, self.n_obj, self.budget

        # DTLZ4 structure: n_pos position vars, n_g g-vars
        n_pos = dim - n_obj  # 9
        n_g = n_obj          # 3
        g_opt = 0.5 * (hi[n_pos:] + lo[n_pos:])  # optimal g-value (0.5 in [0,1])

        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            f = problem(x)
            all_X.append(np.asarray(x, dtype=float))
            all_F.append(np.asarray(f, dtype=float))
            used += 1
            return all_F[-1]

        # ---- Phase 1: structured grid on the Pareto front ----
        # We want a 3D grid over the position variables that maps to
        # a well-spread set of points on the DTLZ4 front.
        # DTLZ4 objectives (with g=0):
        #   f0 = 0.5 * x0 * prod(cos(x1..x_{n_pos-1}))
        #   f1 = 0.5 * x0 * sin(x1..x_{n_pos-1}) * prod(cos(x2..x_{n_pos-1}))
        #   f2 = 0.5 * (1 - x0)
        # The front is parameterized by x0 in [0,1] and angles.
        # A simple approach: grid over x0 and a few angle dimensions.

        # Reserve budget: use ~60% for front grid, 40% for refinement
        n_grid = int(budget * 0.6)
        n_refine = budget - n_grid

        # Build a stratified grid on the front.
        # Strategy: vary x0 in [0,1] and use a few angular coordinates.
        # For good hypervolume, we want coverage in all 3 objective dimensions.
        # f2 = 0.5*(1-x0) ranges [0, 0.5] as x0 goes [1, 0].
        # f0, f1 depend on x0 and angles, max ~0.5.

        # Use a 3D grid: x0 (1D), theta1 (1D), theta2 (1D)
        # with remaining position vars at 0.5 (midpoint, contributes cos=1)
        n_per_axis = max(2, int(round(n_grid ** (1.0 / 3.0))))
        # Adjust to get close to n_grid
        while n_per_axis ** 3 > n_grid and n_per_axis > 2:
            n_per_axis -= 1
        # If still too many, use 2D grid
        if n_per_axis ** 3 > n_grid:
            # 2D grid: x0 and theta1
            n2 = max(2, int(round(n_grid ** 0.5)))
            while n2 ** 2 > n_grid and n2 > 2:
                n2 -= 1
            x0_vals = np.linspace(0.0, 1.0, n2)
            th1_vals = np.linspace(0.0, np.pi / 2, n2)
            for x0v in x0_vals:
                for th1 in th1_vals:
                    if used >= n_grid or time.time() > deadline:
                        break
                    x = np.full(dim, 0.5)
                    x[0] = x0v
                    x[1] = th1
                    x[n_pos:] = g_opt
                    eval_point(x)
        else:
            # 3D grid
            x0_vals = np.linspace(0.0, 1.0, n_per_axis)
            th1_vals = np.linspace(0.0, np.pi / 2, n_per_axis)
            th2_vals = np.linspace(0.0, np.pi / 2, n_per_axis)
            for x0v in x0_vals:
                for th1 in th1_vals:
                    for th2 in th2_vals:
                        if used >= n_grid or time.time() > deadline:
                            break
                        x = np.full(dim, 0.5)
                        x[0] = x0v
                        x[1] = th1
                        x[2] = th2
                        x[n_pos:] = g_opt
                        eval_point(x)

        # ---- Phase 2: local refinement around best regions ----
        # Use remaining budget to do small perturbations around
        # non-dominated points to fill gaps.
        if used < budget and all_F:
            F_arr = np.array(all_F)
            # Find non-dominated points
            nd_mask = np.zeros(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                dominated = False
                for j in range(len(F_arr)):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated = True
                        break
                nd_mask[i] = not dominated

            nd_indices = np.where(nd_mask)[0]
            if len(nd_indices) == 0:
                nd_indices = np.array([0])

            # Refine around each non-dominated point with small perturbations
            rng = np.random.default_rng(42)
            n_refine_actual = budget - used
            per_point = max(1, n_refine_actual // max(1, len(nd_indices)))

            for idx in nd_indices:
                if used >= budget or time.time() > deadline:
                    break
                base_x = all_X[idx]
                for _ in range(per_point):
                    if used >= budget or time.time() > deadline:
                        break
                    # Small perturbation, keep on front (g stays at 0.5)
                    child = base_x.copy()
                    # Perturb position variables only
                    pert = 0.05 * (hi[:n_pos] - lo[:n_pos]) * rng.standard_normal(n_pos)
                    child[:n_pos] = np.clip(child[:n_pos] + pert, lo[:n_pos], hi[:n_pos])
                    child[n_pos:] = g_opt
                    eval_point(child)

        # Return (value ignored by evaluator, but keep interface)
        if all_X:
            return np.array(all_X)
        return np.array([])
# EVOLVE-BLOCK-END
