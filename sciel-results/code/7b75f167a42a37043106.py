"""Multi-objective optimizer for DTLZ7 exploiting known front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit DTLZ7 structure: front determined by (x0, x1); drive g->0."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ7 structure:
        # f1 = x0
        # f2 = (1-x0)*cos(x1*pi/2)
        # f3 = (1-x0)*sin(x1*pi/2)
        # g = 1 + 10*(dim-2) + sum_{i=2}^{dim-1}(x_i - 5*sin(2*pi*x_i))^2
        # F2 = (1+g)*f2, F3 = (1+g)*f3
        #
        # On the Pareto front, g = g_min (minimum possible value of g)
        # g_min = 1 + 10*(dim-2) when all x_i (i>=2) are at their optimal values
        # For dim=22, g_min = 1 + 10*20 = 201
        #
        # The Pareto front in objective space is parameterized by (x0, x1):
        # F1 = x0
        # F2 = (1+g_min)*(1-x0)*cos(x1*pi/2)
        # F3 = (1+g_min)*(1-x0)*sin(x1*pi/2)
        #
        # We need to find x[2:] that minimizes g, then sample (x0, x1) to cover the front.

        # Phase 1: Find a configuration of x[2:] that minimizes g
        # The term (x - 5*sin(2*pi*x))^2 has minima at x=0, 0.5, 1 (value 0)
        # Let's try a few configurations and pick the best

        # Configuration 1: all zeros
        x_test1 = np.zeros(dim)
        f_test1 = problem(x_test1)
        used = 1

        # Configuration 2: all 0.5
        x_test2 = np.zeros(dim)
        x_test2[2:] = 0.5
        f_test2 = problem(x_test2)
        used += 1

        # Configuration 3: all ones
        x_test3 = np.zeros(dim)
        x_test3[2:] = 1.0
        f_test3 = problem(x_test3)
        used += 1

        # Pick the one with lowest sum of objectives (proxy for lowest g)
        candidates = [(x_test1, f_test1), (x_test2, f_test2), (x_test3, f_test3)]
        best_g_idx = np.argmin([np.sum(f) for _, f in candidates])
        x_best_g = candidates[best_g_idx][0].copy()
        f_best_g = candidates[best_g_idx][1].copy()

        # Try to further improve g by local optimization of x[2:]
        # Use a simple coordinate descent
        improved = True
        max_iter = 50
        step = 0.1
        for it in range(max_iter):
            if used >= budget or time.time() > deadline:
                break
            improved = False
            for i in range(2, dim):
                # Try moving x[i] up and down
                for delta in [step, -step]:
                    x_trial = x_best_g.copy()
                    x_trial[i] = np.clip(x_best_g[i] + delta, lo[i], hi[i])
                    f_trial = problem(x_trial)
                    used += 1
                    if np.sum(f_trial) < np.sum(f_best_g):
                        x_best_g = x_trial
                        f_best_g = f_trial
                        improved = True
                        break
                if used >= budget or time.time() > deadline:
                    break
            if not improved:
                # Try smaller steps
                step *= 0.5
                if step < 1e-4:
                    break

        # Now we have a good x[2:] configuration. Extract g_min from this.
        # From f_best_g: F1 = x0 = 0 (since x_best_g[0]=0)
        # F2 = (1+g)*f2, F3 = (1+g)*f3
        # With x0=0: f2 = cos(x1*pi/2), f3 = sin(x1*pi/2)
        # x_best_g[1] = 0, so f2 = 1, f3 = 0
        # Thus F2 = 1+g, F3 = 0
        # So g = F2 - 1
        g_est = f_best_g[1] - 1.0 if f_best_g[1] > 0 else 0.0
        one_plus_g = 1.0 + g_est

        # Phase 2: Sample the Pareto front by varying (x0, x1)
        # The front is: F1 = x0, F2 = (1+g_min)*(1-x0)*cos(x1*pi/2), F3 = (1+g_min)*(1-x0)*sin(x1*pi/2)
        # We need to choose x0, x1 to maximize hypervolume
        #
        # Strategy: Use a stratified grid in (x0, x1) space, but optimize the spacing
        # for hypervolume. The reference point is [1.1, 1.1, 25.0].
        #
        # Key insight: The Pareto front is a 2D manifold. We want to place points
        # to maximize the volume they enclose below the front and above the reference.
        #
        # Let's use a combination of:
        # 1. A dense grid covering the front
        # 2. Additional points at critical locations (corners, edges)

        # Determine how many evaluations we have left
        remaining = budget - used
        if remaining <= 0:
            return np.array([x_best_g])

        # Allocate evaluations:
        # - Most to grid sampling of (x0, x1)
        # - Some to verifying/improving g

        # Grid size: aim for ~sqrt(remaining) points in each dimension
        # But we need to be strategic about placement

        # Let's use a stratified sampling approach
        # The front in (F1, F2, F3) space:
        # F1 ranges from 0 to 1
        # For each F1=x0, F2 and F3 trace a quarter circle of radius (1+g_min)*(1-x0)
        #
        # To maximize hypervolume, we want to:
        # 1. Cover the full range of F1 (0 to 1)
        # 2. For each F1, cover the quarter circle in (F2, F3)
        #
        # Optimal spacing for hypervolume on a smooth front:
        # Uniform in the parameter space tends to work well

        # Let's create a grid of (x0, x1) values
        # x0 in [0, 1], x1 in [0, pi/2]
        #
        # Number of points: let's use n0 x n1 grid
        n0 = int(np.sqrt(remaining * 0.6))  # 60% of remaining for grid
        n1 = int(remaining * 0.6 / max(n0, 1))
        n0 = max(n0, 2)
        n1 = max(n1, 2)

        # Ensure we don't exceed budget
        n_points = n0 * n1
        if n_points > remaining:
            # Reduce grid size
            scale = np.sqrt(remaining / n_points)
            n0 = max(int(n0 * scale), 1)
            n1 = max(int(n1 * scale), 1)
            n_points = n0 * n1

        # Create stratified grid
        # Use midpoints of intervals for better coverage
        x0_vals = (np.arange(n0) + 0.5) / n0
        x1_vals = (np.arange(n1) + 0.5) / n1 * (np.pi / 2)

        # Add boundary points for better coverage
        x0_vals = np.concatenate([x0_vals, [0.0, 1.0]])
        x1_vals = np.concatenate([x1_vals, [0.0, np.pi/2]])

        # Now evaluate all grid points
        # x[2:] is fixed at x_best_g[2:]
        all_x = []
        all_f = []

        for x0 in x0_vals:
            for x1 in x1_vals:
                if used >= budget or time.time() > deadline:
                    break
                x = x_best_g.copy()
                x[0] = np.clip(x0, lo[0], hi[0])
                x[1] = np.clip(x1, lo[1], hi[1])
                f = problem(x)
                used += 1
                all_x.append(x)
                all_f.append(f)
            if used >= budget or time.time() > deadline:
                break

        # Phase 3: Use remaining budget to refine g or add more front points
        remaining = budget - used
        if remaining > 0 and time.time() < deadline:
            # Try to improve g further with more local search
            step = 0.01
            for it in range(min(remaining, 20)):
                if used >= budget or time.time() > deadline:
                    break
                improved = False
                for i in range(2, dim):
                    if used >= budget or time.time() > deadline:
                        break
                    for delta in [step, -step]:
                        if used >= budget:
                            break
                        x_trial = x_best_g.copy()
                        x_trial[i] = np.clip(x_best_g[i] + delta, lo[i], hi[i])
                        f_trial = problem(x_trial)
                        used += 1
                        if np.sum(f_trial) < np.sum(f_best_g):
                            x_best_g = x_trial
                            f_best_g = f_trial
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-5:
                        break

            # Add more points along the front with improved g
            remaining = budget - used
            if remaining > 0 and time.time() < deadline:
                # Re-evaluate some grid points with improved x[2:]
                n_extra = min(remaining, 30)
                for i in range(n_extra):
                    if used >= budget or time.time() > deadline:
                        break
                    x0 = np.random.uniform(0, 1)
                    x1 = np.random.uniform(0, np.pi/2)
                    x = x_best_g.copy()
                    x[0] = np.clip(x0, lo[0], hi[0])
                    x[1] = np.clip(x1, lo[1], hi[1])
                    f = problem(x)
                    used += 1
                    all_x.append(x)
                    all_f.append(f)

        if len(all_x) == 0:
            return np.array([x_best_g])

        return np.array(all_x)
# EVOLVE-BLOCK-END
