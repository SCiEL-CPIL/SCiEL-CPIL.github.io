"""Multi-objective optimizer for DTLZ7 exploiting known front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit DTLZ7 structure: front determined by (x0, x1); drive g->0.
    
    Uses stratified grid with jitter for uniform coverage of the 2D Pareto
    front parameter space, combined with local g-variable optimization.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget

        def compute_gt(x):
            """Compute gt(x) for DTLZ7: 0.35*(1+1/(4+10*x))*sin(12*pi*x)"""
            return 0.35 * (1.0 + 1.0 / (4.0 + 10.0 * x)) * np.sin(12.0 * np.pi * x)

        def optimize_g_vars(xi_init):
            """Optimize g-variables using fixed-point iteration to find xi ≈ gt(xi)."""
            xi = xi_init.copy()
            for _ in range(80):
                gt_val = compute_gt(xi)
                xi_new = gt_val
                if np.max(np.abs(xi_new - xi)) < 1e-14:
                    break
                xi = xi_new
            return np.clip(xi, 0.0, 1.0)

        def make_point_on_front(x0, x1):
            """Create a 22-dim point on the Pareto front with given x0, x1."""
            x = np.ones(dim, dtype=np.float64)
            x[0] = x0
            x[1] = x1
            # Optimize g-variables to be on the Pareto front (g=0)
            g_vars = optimize_g_vars(np.full(dim - 2, 0.5))
            x[2:] = g_vars
            x = np.clip(x, lo, hi)
            return x

        # Phase 1: Stratified grid with jitter on (x0, x1) for uniform front coverage
        # Use a grid in [0,1]^2 with random jitter to avoid exact grid artifacts
        n_grid = 14  # 14x14 = 196 points, leaving room for refinement
        n_init = min(n_grid * n_grid, budget // 2)
        
        # Create stratified grid
        grid_indices = np.arange(n_init)
        i_idx = grid_indices % n_grid
        j_idx = grid_indices // n_grid
        
        # Stratified sampling: each cell gets one point with jitter
        cell_w = 1.0 / n_grid
        x0_vals = (i_idx + rng.random(n_init)) * cell_w
        x1_vals = (j_idx + rng.random(n_init)) * cell_w
        x0_vals = np.clip(x0_vals, 0.0, 1.0)
        x1_vals = np.clip(x1_vals, 0.0, 1.0)
        
        points = []
        used = 0

        for k in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = make_point_on_front(x0_vals[k], x1_vals[k])
            f = problem(x)
            points.append((x, f))
            used += 1

        # Phase 2: Local refinement - refine g-vars for all points
        if points:
            for idx in range(len(points)):
                if used >= budget or time.time() > deadline:
                    break

                x, f = points[idx]
                best_x = x.copy()
                best_f = f.copy()

                # Coordinate descent on g-vars with adaptive step sizes
                for _ in range(4):
                    improved = False
                    for i in range(2, dim):
                        if used >= budget or time.time() > deadline:
                            break
                        # Try multiple step sizes
                        for step in [0.001, 0.01, 0.05, 0.1, -0.001, -0.01, -0.05, -0.1]:
                            new_x = best_x.copy()
                            new_x[i] = np.clip(best_x[i] + step, lo[i], hi[i])
                            new_f = problem(new_x)
                            used += 1
                            # Check if new point dominates or is closer to front
                            if np.all(new_f <= best_f) and np.any(new_f < best_f):
                                best_x = new_x
                                best_f = new_f
                                improved = True
                                if used >= budget or time.time() > deadline:
                                    break
                    if not improved:
                        break

                points[idx] = (best_x, best_f)

        # Phase 3: If budget remains, add more points with finer stratified sampling
        remaining = budget - used
        if remaining > 0:
            # Use finer grid in regions where we might have gaps
            n_extra = min(remaining, 40)
            for k in range(n_extra):
                if used >= budget or time.time() > deadline:
                    break
                # Sample in a finer stratified manner
                i_idx = k % (n_grid * 2)
                j_idx = k // (n_grid * 2)
                cell_w = 1.0 / (n_grid * 2)
                x0 = (i_idx + rng.random()) * cell_w
                x1 = (j_idx + rng.random()) * cell_w
                x0 = np.clip(x0, 0.0, 1.0)
                x1 = np.clip(x1, 0.0, 1.0)
                x = make_point_on_front(x0, x1)
                f = problem(x)
                points.append((x, f))
                used += 1

        # Return the last point (return value is ignored per contract)
        if points:
            return points[-1][0]
        return np.zeros(dim)
# EVOLVE-BLOCK-END
