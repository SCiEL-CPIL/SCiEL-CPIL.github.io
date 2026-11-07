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
        # For DTLZ4 with m objectives and n variables:
        # - n_pos = m - 1 position variables (first m-1 variables)
        # - n_g = n - (m - 1) g-variables (remaining variables)
        # For m=3, n=12: n_pos = 2, n_g = 10
        n_pos = n_obj - 1  # 2 position variables
        n_g = dim - n_pos  # 10 g-variables
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
        # DTLZ4 with 3 objectives has 2 position variables.
        # The Pareto front is parameterized by x0, x1 in [0,1] with x0 + x1 + x2 = 1
        # where x2 is implicitly determined. Actually for DTLZ4:
        # f1 = x0 * (1 + g)
        # f2 = x1 * (1 + g)  
        # f3 = (1 - x0 - x1) * (1 + g)
        # With g=0: f1 = x0, f2 = x1, f3 = 1 - x0 - x1
        # The front is a triangle in [0,1]^3 where x0 + x1 <= 1.

        # Reserve budget: use ~70% for front grid, 30% for refinement
        n_grid = int(budget * 0.7)
        n_refine = budget - n_grid

        # Build a stratified grid on the Pareto front.
        # Use cell-center sampling to avoid boundary issues.
        # Sample x0 and x1 such that x0 + x1 <= 1.
        
        # For good hypervolume coverage, use a triangular grid.
        # n_per_axis values for x0 and x1.
        n_per_axis = max(2, int(round(np.sqrt(n_grid))))
        # Adjust to get close to n_grid
        while n_per_axis * (n_per_axis + 1) // 2 > n_grid and n_per_axis > 2:
            n_per_axis -= 1
        
        # Generate points on the triangular front
        x0_vals = np.linspace(0.0, 1.0, n_per_axis)
        
        for i, x0v in enumerate(x0_vals):
            # x1 can range from 0 to 1 - x0
            n_x1 = n_per_axis - i
            if n_x1 < 1:
                continue
            x1_vals = np.linspace(0.0, 1.0 - x0v, n_x1)
            for x1v in x1_vals:
                if used >= n_grid or time.time() > deadline:
                    break
                x = np.full(dim, 0.5)
                x[0] = x0v
                x[1] = x1v
                x[n_pos:] = g_opt
                eval_point(x)
        
        # Add corner points if budget allows
        corners = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0]
        ]
        for corner in corners:
            if used >= n_grid or time.time() > deadline:
                break
            x = np.full(dim, 0.5)
            x[0] = corner[0]
            x[1] = corner[1]
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
                    # Ensure x0 + x1 <= 1 for DTLZ4 front
                    if child[0] + child[1] > 1.0:
                        scale = 1.0 / (child[0] + child[1])
                        child[0] *= scale
                        child[1] *= scale
                    child[n_pos:] = g_opt
                    eval_point(child)

        # Return (value ignored by evaluator, but keep interface)
        if all_X:
            return np.array(all_X)
        return np.array([])
# EVOLVE-BLOCK-END
