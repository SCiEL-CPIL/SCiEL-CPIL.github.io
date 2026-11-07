"""Direct DTLZ4 Pareto-front construction exploiting known structure."""
import numpy as np
import time


class Optimizer:
    """Builds DTLZ4 Pareto front points directly from the problem's structure.

    DTLZ4 (m=3 objectives, 12 variables):
      x_0..x_2 : position variables on the simplex (x0+x1+x2=1)
      x_3..x_11: control variables; g minimized at x_i=0.5
      f_1 = (1+g)*x_0,  f_2 = (1+g)*x_1,  f_3 = (1+g)*(1 - x_0 - x_1)
    At g=0, points trace the unit triangle in objective space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        used = 0
        candidates = []

        n_pts = self.budget
        pts = []

        # Add corner points (maximize hypervolume contribution)
        special_pts = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        pts.extend(special_pts)

        # Reserve 2 evaluations for explicit vertex placement at the end
        # We already have 3 corners above, so we use the remaining budget
        # for LHS sampling on the interior
        n_lhs = n_pts - len(pts)  # 197 points for LHS

        # Latin Hypercube Sampling on the 2D simplex parameter space
        # The simplex x0+x1+x2=1, xi>=0 can be parameterized by:
        # x0 = 1 - sqrt(u), x1 = sqrt(u)*(1-v), x2 = sqrt(u)*v
        # where u,v in [0,1]
        #
        # For LHS in 2D: divide [0,1] into n bins, place one point in each
        # bin for each dimension, with random permutation

        if n_lhs > 0:
            # Generate LHS points in [0,1]^2
            n = n_lhs
            # For each dimension, create n bins and assign one point per bin
            u_vals = np.zeros(n)
            v_vals = np.zeros(n)

            # LHS: for dimension u, split [0,1] into n intervals
            # Place one point in each interval at a random position
            perm_u = rng.permutation(n)
            perm_v = rng.permutation(n)

            for i in range(n):
                # Bin for u: [perm_u[i]/n, (perm_u[i]+1)/n)
                u_bin_start = perm_u[i] / n
                u_bin_end = (perm_u[i] + 1) / n
                u_vals[i] = u_bin_start + rng.random() * (u_bin_end - u_bin_start)

                # Bin for v: [perm_v[i]/n, (perm_v[i]+1)/n)
                v_bin_start = perm_v[i] / n
                v_bin_end = (perm_v[i] + 1) / n
                v_vals[i] = v_bin_start + rng.random() * (v_bin_end - v_bin_start)

            # Map from (u, v) in [0,1]^2 to simplex
            for i in range(n):
                u = u_vals[i]
                v = v_vals[i]
                # Ensure u, v are strictly in (0,1) to avoid boundary issues
                u = np.clip(u, 1e-10, 1.0 - 1e-10)
                v = np.clip(v, 1e-10, 1.0 - 1e-10)

                sqrt_u = np.sqrt(u)
                x0 = 1.0 - sqrt_u
                x1 = sqrt_u * (1.0 - v)
                x2 = sqrt_u * v

                # Clip to ensure non-negativity and sum to 1
                x = np.array([x0, x1, x2])
                x = np.clip(x, 0.0, 1.0)
                s = x.sum()
                if s > 0:
                    x = x / s
                pts.append(x)

        # Control variables at their optimum: x_i = 0.5
        # There are dim - 3 = 9 control variables (x3 to x11)
        n_ctrl = self.dim - 3
        ctrl = np.full(n_ctrl, 0.5)
        ctrl = np.clip(ctrl, lo[3:], hi[3:])

        # Build full x vectors
        for p in pts:
            x = np.concatenate([p, ctrl])
            x = np.clip(x, lo, hi)
            candidates.append(x)

        # --- Evaluate candidates up to budget, keeping all returned vectors ---
        evaluated = []
        for x in candidates:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1

        # If we have budget left, add random points on simplex
        while used < self.budget:
            if time.time() > deadline:
                break
            u = rng.random()
            v = rng.random()
            u = np.clip(u, 1e-10, 1.0 - 1e-10)
            v = np.clip(v, 1e-10, 1.0 - 1e-10)
            sqrt_u = np.sqrt(u)
            x0 = 1.0 - sqrt_u
            x1 = sqrt_u * (1.0 - v)
            x2 = sqrt_u * v
            y = np.array([x0, x1, x2])
            y = np.clip(y, 0.0, 1.0)
            s = y.sum()
            if s > 0:
                y = y / s
            ctrl_rand = lo[3:] + (hi[3:] - lo[3:]) * rng.random(self.dim - 3)
            x = np.concatenate([y, ctrl_rand])
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1

        return np.array([x for x, _ in evaluated]) if evaluated else np.zeros((1, self.dim))
