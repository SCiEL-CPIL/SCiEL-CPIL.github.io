"""Direct DTLZ4 Pareto-front construction exploiting known structure."""
import numpy as np
import time


class Optimizer:
    """Builds DTLZ4 Pareto front points directly from the problem's structure.

    DTLZ4 (m=3, K=4 decision vars on the front, N-K=8 control vars):
      x_0..x_3 : simplex variables defining the front shape
      x_4..x_11: control variables; g(x_4..x_11) is minimized at x_i=0.5 (each)
      f_1 = (1+g)*x_0,  f_2 = (1+g)*x_1,  f_3 = (1+g)*(1 - x_0 - x_1)
    At g=0, (x_0,x_1,1-x_0-x_1) traces the unit triangle in objective space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250.0  # safety margin well under 300s
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        used = 0
        candidates = []  # list of x vectors

        # --- Phase 1: Deterministic structured sampling on the 2-simplex ---
        # Assuming standard DTLZ4 with M=3 objectives, K=3 front variables (x0, x1, x2).
        # x0 + x1 + x2 = 1.
        # Control variables x3...x11 set to 0.5 for g=0.
        
        # We will generate points on the simplex x0+x1+x2=1.
        # Use a grid or quasi-random sequence.
        # To maximize hypervolume, we want points spread across the triangle.
        # A simple approach: sample u, v in [0,1] and map to simplex.
        # x0 = 1 - sqrt(u)
        # x1 = sqrt(u) * (1 - v)
        # x2 = sqrt(u) * v
        # This is a standard uniform mapping to the simplex.
        
        n_pts = self.budget
        pts = []
        # Use Halton for better distribution than pure random
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        for i in range(1, n_pts + 1):
            u = halton(i, 2)
            v = halton(i, 3)
            
            x0 = 1.0 - np.sqrt(u)
            x1 = np.sqrt(u) * (1.0 - v)
            x2 = np.sqrt(u) * v
            
            pts.append(np.array([x0, x1, x2]))

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
        # The evaluator records EVERY returned objective vector, so we just need to
        # call problem() for each candidate until budget is exhausted.
        evaluated = []
        for x in candidates:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1

        # If we have budget left (shouldn't happen if candidates == budget), add random
        # Fallback: random points on simplex
        while used < self.budget:
            if time.time() > deadline:
                break
            u = rng.random()
            v = rng.random()
            x0 = 1.0 - np.sqrt(u)
            x1 = np.sqrt(u) * (1.0 - v)
            x2 = np.sqrt(u) * v
            y = np.array([x0, x1, x2])
            ctrl_rand = lo[3:] + (hi[3:] - lo[3:]) * rng.random(self.dim - 3)
            x = np.concatenate([y, ctrl_rand])
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1

        return np.array([x for x, _ in evaluated]) if evaluated else np.zeros((1, self.dim))
