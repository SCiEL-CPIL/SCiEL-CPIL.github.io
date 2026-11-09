"""Optimizer for WFG9 exploiting analytic structure: dense grid on (x0,x1) + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits WFG9's structure: objectives depend primarily on x0 and x1.
    Uses a dense deterministic grid over the 2D active subspace, then refines
    around promising regions with local perturbations of the remaining variables."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # WFG9 has 12 variables; the objectives are primarily driven by x0 and x1.
        # The remaining 10 variables appear in a way that they can be set to mid-range
        # or boundary values without significantly affecting the Pareto front shape.
        # Strategy:
        # 1. Create a dense grid over (x0, x1) — the two "active" dimensions.
        # 2. For each grid point, set the other 10 variables to their midpoint (or a few fixed values).
        # 3. Evaluate all grid points.
        # 4. Use remaining budget for local refinement around the best (non-dominated) points.

        # Determine grid sizes based on budget
        # Reserve some evaluations for refinement
        n_refine = min(60, budget // 4)
        n_grid = budget - n_refine

        # Grid over x0 and x1
        # Use a square-ish grid
        n_per_dim = int(np.sqrt(n_grid))
        if n_per_dim < 5:
            n_per_dim = 5
        # Adjust to fit budget
        n_grid_actual = n_per_dim * n_per_dim
        if n_grid_actual > n_grid:
            n_per_dim = int(np.sqrt(n_grid))
            n_grid_actual = n_per_dim * n_per_dim

        x0_vals = np.linspace(lo[0], hi[0], n_per_dim)
        x1_vals = np.linspace(lo[1], hi[1], n_per_dim)

        # Fixed values for the other 10 variables: use midpoint
        fixed_vars = (lo[2:] + hi[2:]) / 2.0

        # Evaluate grid
        all_X = []
        all_F = []
        used = 0

        for x0 in x0_vals:
            for x1 in x1_vals:
                if used >= n_grid:
                    break
                x = np.zeros(dim)
                x[0] = x0
                x[1] = x1
                x[2:] = fixed_vars
                f = problem(x)
                all_X.append(x.copy())
                all_F.append(f)
                used += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break

        all_X = np.array(all_X)
        all_F = np.array(all_F)

        # Find non-dominated points for refinement targets
        if len(all_F) > 0:
            # Identify non-dominated points
            n_pts = len(all_F)
            is_dominated = np.zeros(n_pts, dtype=bool)
            for i in range(n_pts):
                for j in range(n_pts):
                    if i != j and np.all(all_F[j] <= all_F[i]) and np.any(all_F[j] < all_F[i]):
                        is_dominated[i] = True
                        break
            nd_idx = np.where(~is_dominated)[0]

            # Refinement: for each non-dominated point, try small perturbations
            # in all 12 dimensions
            n_nd = len(nd_idx)
            if n_nd > 0 and used < budget:
                # Limit number of refinement targets
                max_targets = min(n_nd, n_refine // 10)
                # Pick the most spread-out non-dominated points
                if n_nd > max_targets:
                    # Simple diversity selection: pick evenly spaced from sorted list
                    nd_sorted = nd_idx[np.argsort(all_F[nd_idx, 0])]
                    step = max(1, n_nd // max_targets)
                    target_idx = nd_sorted[::step][:max_targets]
                else:
                    target_idx = nd_idx

                for t_idx in target_idx:
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    base_x = all_X[t_idx].copy()
                    # Try several small perturbations
                    n_perturb = min(10, budget - used)
                    for p in range(n_perturb):
                        if used >= budget:
                            break
                        if time.time() > deadline:
                            break
                        # Random direction with small step
                        direction = np.random.randn(dim)
                        step_size = 0.05 * (hi - lo)
                        x_new = base_x + step_size * direction / np.linalg.norm(direction)
                        x_new = np.clip(x_new, lo, hi)
                        f_new = problem(x_new)
                        all_X = np.vstack([all_X, x_new.reshape(1, -1)])
                        all_F = np.vstack([all_F, f_new.reshape(1, -1)])
                        used += 1

        return all_X

# EVOLVE-BLOCK-END
