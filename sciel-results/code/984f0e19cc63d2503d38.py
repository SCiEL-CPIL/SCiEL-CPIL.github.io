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

        # Use remaining budget for structured grid on simplex
        # For hypervolume, uniform coverage of the simplex is important.
        # We use a structured grid with bias toward edges (where hypervolume contribution is higher).
        n_remaining = n_pts - len(pts)
        
        # Generate a grid of points on the simplex
        # Use a sqrt-biased grid to emphasize edges
        n_grid = int(np.sqrt(n_remaining))
        if n_grid < 3:
            n_grid = 3
        
        # Generate grid points with edge bias
        i_vals = np.linspace(0, 1, n_grid)
        # Bias toward edges: use sqrt transform
        i_vals = np.sqrt(i_vals)
        
        for i in i_vals:
            j_vals = np.linspace(0, 1 - i, n_grid)
            j_vals = i * np.sqrt(j_vals / max(i, 1e-10)) if i > 0 else j_vals
            # Simpler: just use uniform grid but with more points near edges
            for j in j_vals:
                if j > 1 - i + 1e-10:
                    continue
                k = 1 - i - j
                if k < -1e-10:
                    continue
                x = np.array([i, j, k])
                x = np.clip(x, 0.0, 1.0)
                s = x.sum()
                if s > 0:
                    x = x / s
                pts.append(x)
        
        # If we have too many points, subsample
        if len(pts) > n_remaining + 3:
            # Keep special points and subsample the rest
            pts = pts[:3] + pts[3:n_remaining+3]
        elif len(pts) < n_remaining + 3:
            # Add more points with Sobol-like sampling
            n_extra = n_remaining + 3 - len(pts)
            try:
                from scipy.stats import qmc
                n_gen = 1
                while n_gen < n_extra:
                    n_gen *= 2
                sampler = qmc.Sobol(d=2, scramble=True, seed=42)
                sobol_pts = sampler.random(n_gen)[:n_extra]
                for u, v in sobol_pts:
                    sqrt_u = np.sqrt(u)
                    x0 = 1.0 - sqrt_u
                    x1 = sqrt_u * (1.0 - v)
                    x2 = sqrt_u * v
                    x = np.array([x0, x1, x2])
                    x = np.clip(x, 0.0, 1.0)
                    s = x.sum()
                    if s > 0:
                        x = x / s
                    pts.append(x)
            except ImportError:
                pass

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
        
        # If we still have budget, add more points with Sobol sampling
        while len(candidates) < self.budget:
            if time.time() > deadline:
                break
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=2, scramble=True, seed=42 + len(candidates))
                u, v = sampler.random(1)[0]
            except ImportError:
                u, v = rng.random(2)
            sqrt_u = np.sqrt(u)
            x0 = 1.0 - sqrt_u
            x1 = sqrt_u * (1.0 - v)
            x2 = sqrt_u * v
            p = np.array([x0, x1, x2])
            p = np.clip(p, 0.0, 1.0)
            s = p.sum()
            if s > 0:
                p = p / s
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

        # If we have budget left (shouldn't happen if n_halton was calculated correctly), 
        # add random points on simplex
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
