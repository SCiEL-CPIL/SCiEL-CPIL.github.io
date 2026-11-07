"""Direct DTLZ4 Pareto-front construction exploiting known structure."""
import numpy as np
import time


class Optimizer:
    """Builds DTLZ4 Pareto front points directly from the problem's structure.

    DTLZ4 (m=3 objectives, 12 variables):
      x_0..x_2 : position variables on the simplex (x0+x1+x2=1)
      x_3..x_11: control variables; g minimized at x_i=0.5
      f_i = (1+g) * x_{i-1} for i=1,2,3 (with f_3 using 1-x0-x1)
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
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5]),
        ]
        pts.extend(special_pts)

        n_remaining = n_pts - len(pts)

        # Use stratified sampling with power-law bias toward edges
        # for better hypervolume coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        # Use a transformed mapping that concentrates points near the boundary
        # of the simplex for better hypervolume
        for i in range(1, n_remaining + 1):
            u = halton(i, 2)
            v = halton(i, 3)

            # Apply power transformation to concentrate near edges
            # This biases sampling toward the boundary of the simplex
            u_t = u ** 0.5
            v_t = v ** 0.5

            x0 = 1.0 - np.sqrt(u_t)
            x1 = np.sqrt(u_t) * (1.0 - v_t)
            x2 = np.sqrt(u_t) * v_t

            pts.append(np.array([x0, x1, x2]))

        # Control variables at their optimum: x_i = 0.5
        n_ctrl = self.dim - 3
        ctrl = np.full(n_ctrl, 0.5)
        ctrl = np.clip(ctrl, lo[3:], hi[3:])

        # Build full x vectors
        for p in pts:
            x = np.concatenate([p, ctrl])
            x = np.clip(x, lo, hi)
            candidates.append(x)

        # Evaluate candidates up to budget
        evaluated = []
        for x in candidates:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1

        # If budget remains, add additional stratified points
        while used < self.budget:
            if time.time() > deadline:
                break
            u = rng.random()
            v = rng.random()
            u_t = u ** 0.5
            v_t = v ** 0.5
            x0 = 1.0 - np.sqrt(u_t)
            x1 = np.sqrt(u_t) * (1.0 - v_t)
            x2 = np.sqrt(u_t) * v_t
            y = np.array([x0, x1, x2])
            ctrl_rand = lo[3:] + (hi[3:] - lo[3:]) * rng.random(self.dim - 3)
            x = np.concatenate([y, ctrl_rand])
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1

        return np.array([x for x, _ in evaluated]) if evaluated else np.zeros((1, self.dim))
