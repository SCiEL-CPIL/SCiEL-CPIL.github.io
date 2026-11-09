"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """WFG9 optimizer leveraging analytic structure: identify active dims, dense grid, local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        span = hi - lo
        
        used = 0
        all_F = []
        
        def eval_and_record(x):
            nonlocal used
            if time.time() > deadline:
                return None
            f = np.array(problem(x), dtype=float)
            used += 1
            all_F.append(f)
            return f
        
        # Phase 1: Probe which variables affect objectives (use ~10% budget)
        probe_budget = max(10, self.budget // 10)
        base = mid.copy()
        probe_results = {}  # var_idx -> list of objective changes
        
        for i in range(self.dim):
            if used + 2 > self.budget:
                break
            x_plus = base.copy()
            x_plus[i] = hi[i]
            x_minus = base.copy()
            x_minus[i] = lo[i]
            fp = eval_and_record(x_plus)
            fm = eval_and_record(x_minus)
            if fp is None or fm is None:
                break
            diff = np.max(np.abs(fp - fm))
            probe_results[i] = diff
        
        # Identify active dimensions (those with significant effect)
        if probe_results:
            max_diff = max(probe_results.values())
            if max_diff > 1e-6:
                active_dims = [i for i, d in probe_results.items() if d > 0.01 * max_diff]
            else:
                active_dims = list(range(min(3, self.dim)))
        else:
            active_dims = list(range(min(3, self.dim)))
        
        # Limit active dims to keep grid manageable
        max_active = min(len(active_dims), 4)
        active_dims = active_dims[:max_active]
        
        # Phase 2: Dense grid on active dimensions, fix others to mid
        remaining = self.budget - used
        if remaining < 5:
            return all_F
        
        n_active = len(active_dims)
        
        if n_active == 0:
            # All variables seem inactive; just do random search
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                x = lo + span * rng.random(self.dim)
                eval_and_record(x)
            return all_F
        
        # Determine grid resolution
        # Use most of remaining budget for grid
        grid_budget = int(remaining * 0.7)
        polish_budget = remaining - grid_budget
        
        # For n_active dims, use n^d grid points
        if n_active == 1:
            n_grid = min(grid_budget, 100)
            grids = [np.linspace(lo[active_dims[0]], hi[active_dims[0]], n_grid)]
        elif n_active == 2:
            n_per = max(2, int(np.sqrt(grid_budget)))
            n_per = min(n_per, 50)
            grids = [np.linspace(lo[active_dims[0]], hi[active_dims[0]], n_per),
                     np.linspace(lo[active_dims[1]], hi[active_dims[1]], n_per)]
        elif n_active == 3:
            n_per = max(2, int(round(grid_budget ** (1.0/3.0))))
            n_per = min(n_per, 20)
            grids = [np.linspace(lo[active_dims[0]], hi[active_dims[0]], n_per),
                     np.linspace(lo[active_dims[1]], hi[active_dims[1]], n_per),
                     np.linspace(lo[active_dims[2]], hi[active_dims[2]], n_per)]
        else:  # n_active == 4
            n_per = max(2, int(round(grid_budget ** 0.25)))
            n_per = min(n_per, 10)
            grids = [np.linspace(lo[d], hi[d], n_per) for d in active_dims]
        
        # Evaluate grid points
        grid_F = []
        for idxs in np.ndindex(*[len(g) for g in grids]):
            if time.time() > deadline or used >= self.budget:
                break
            x = mid.copy()
            for k, d in enumerate(active_dims):
                x[d] = grids[k][idxs[k]]
            f = eval_and_record(x)
            if f is not None:
                grid_F.append((x.copy(), f))
        
        # Phase 3: Local refinement around best points
        if polish_budget > 5 and grid_F:
            # Find non-dominated points from grid
            grid_F_arr = np.array([f for _, f in grid_F])
            grid_X_arr = np.array([x for x, _ in grid_F])
            
            # Simple non-dominated filter
            n_pts = len(grid_F_arr)
            nd_mask = np.ones(n_pts, dtype=bool)
            for i in range(n_pts):
                if not nd_mask[i]:
                    continue
                for j in range(n_pts):
                    if i == j or not nd_mask[j]:
                        continue
                    if np.all(grid_F_arr[j] <= grid_F_arr[i]) and np.any(grid_F_arr[j] < grid_F_arr[i]):
                        nd_mask[i] = False
                        break
            
            nd_points = [(grid_X_arr[i], grid_F_arr[i]) for i in range(n_pts) if nd_mask[i]]
            
            # Refine around top non-dominated points
            n_refine = min(len(nd_points), 10)
            points_to_refine = nd_points[:n_refine]
            
            for x0, f0 in points_to_refine:
                if used >= self.budget or time.time() > deadline:
                    break
                # Local search with decreasing step
                x_best = x0.copy()
                f_best = f0.copy()
                step = 0.05 * span
                for it in range(20):
                    if used >= self.budget or time.time() > deadline:
                        break
                    # Try perturbations in active dims
                    improved = False
                    for d in active_dims:
                        for direction in [-1, 1]:
                            if used >= self.budget or time.time() > deadline:
                                break
                            x_try = x_best.copy()
                            x_try[d] += direction * step
                            x_try = np.clip(x_try, lo, hi)
                            f_try = eval_and_record(x_try)
                            if f_try is None:
                                continue
                            # Check if it improves in at least one objective without worsening others too much
                            if np.any(f_try < f_best) and np.all(f_try <= f_best + 0.1 * np.max(np.abs(f_best))):
                                x_best = x_try
                                f_best = f_try
                                improved = True
                    if not improved:
                        step *= 0.5
                    else:
                        step *= 0.8
        
        return all_F
# EVOLVE-BLOCK-END
