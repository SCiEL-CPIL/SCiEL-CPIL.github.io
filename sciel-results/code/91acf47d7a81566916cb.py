"""Optimizer for WFG9 exploiting analytic structure: dense grid on (x0,x1) with multiple fixed-value strategies."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits WFG9's structure: objectives depend primarily on x0 and x1.
    Uses a dense grid over the 2D active subspace with multiple fixed-value
    strategies for the remaining 10 variables, then refines around promising
    non-dominated points."""

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
        # The remaining 10 variables affect the shape of the Pareto front.
        # Strategy:
        # 1. Create a dense grid over (x0, x1) — the two "active" dimensions.
        # 2. For each grid point, try multiple fixed-value strategies for the other 10 variables.
        # 3. Evaluate all grid points with all strategies.
        # 4. Use remaining budget for local refinement around the best (non-dominated) points.

        # Determine how many fixed-value strategies to use
        # Each strategy uses a different configuration for variables 2-11
        n_strategies = 3  # midpoint, low-biased, high-biased
        
        # Allocate budget: 70% for grid, 30% for refinement
        n_refine = min(60, budget // 3)
        n_grid_total = budget - n_refine
        
        # Grid over x0 and x1, shared across strategies
        # Use a square-ish grid
        n_per_dim = int(np.sqrt(n_grid_total / n_strategies))
        if n_per_dim < 5:
            n_per_dim = 5
        # Adjust to fit budget
        n_grid_actual = n_per_dim * n_per_dim * n_strategies
        if n_grid_actual > n_grid_total:
            n_per_dim = int(np.sqrt(n_grid_total / n_strategies))
            n_grid_actual = n_per_dim * n_per_dim * n_strategies

        x0_vals = np.linspace(lo[0], hi[0], n_per_dim)
        x1_vals = np.linspace(lo[1], hi[1], n_per_dim)

        # Define multiple fixed-value strategies for variables 2-11
        # Strategy 0: midpoint
        fixed_vars_0 = (lo[2:] + hi[2:]) / 2.0
        # Strategy 1: low-biased (closer to lower bounds)
        fixed_vars_1 = lo[2:] + 0.25 * (hi[2:] - lo[2:])
        # Strategy 2: high-biased (closer to upper bounds)
        fixed_vars_2 = lo[2:] + 0.75 * (hi[2:] - lo[2:])
        fixed_vars_list = [fixed_vars_0, fixed_vars_1, fixed_vars_2]

        # Evaluate grid with all strategies
        all_X = []
        all_F = []
        used = 0

        for strat_idx, fixed_vars in enumerate(fixed_vars_list):
            for x0 in x0_vals:
                for x1 in x1_vals:
                    if used >= n_grid_total:
                        break
                    if time.time() > deadline:
                        break
                    x = np.zeros(dim)
                    x[0] = x0
                    x[1] = x1
                    x[2:] = fixed_vars
                    f = problem(x)
                    all_X.append(x.copy())
                    all_F.append(f)
                    used += 1
                if used >= n_grid_total or time.time() > deadline:
                    break
            if used >= n_grid_total or time.time() > deadline:
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
                max_targets = min(n_nd, max(5, n_refine // 10))
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
                    n_perturb = min(12, budget - used)
                    for p in range(n_perturb):
                        if used >= budget:
                            break
                        if time.time() > deadline:
                            break
                        # Random direction with small step
                        direction = np.random.randn(dim)
                        step_size = 0.03 * (hi - lo)
                        x_new = base_x + step_size * direction / np.linalg.norm(direction)
                        x_new = np.clip(x_new, lo, hi)
                        f_new = problem(x_new)
                        all_X = np.vstack([all_X, x_new.reshape(1, -1)])
                        all_F = np.vstack([all_F, f_new.reshape(1, -1)])
                        used += 1

        return all_X

# EVOLVE-BLOCK-END
