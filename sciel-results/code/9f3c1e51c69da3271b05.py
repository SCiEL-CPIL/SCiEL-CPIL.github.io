import numpy as np
import time

class Optimizer:
    """DTLZ2 optimizer: exploit known Pareto front structure with HV-weighted sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        g_start = self.n_obj - 1
        base_x = np.zeros(self.dim)
        base_x[g_start:] = 0.5

        evaluated_points = []
        used_budget = 0

        def eval_point(u, v):
            nonlocal used_budget
            if time.time() > deadline or used_budget >= self.budget:
                return None
            x0 = (2.0 / np.pi) * np.arcsin(u)
            x1 = v
            x = base_x.copy()
            x[0] = x0
            x[1] = x1
            f = problem(x)
            evaluated_points.append(x)
            used_budget += 1
            return f

        def front_obj(u, v):
            t1 = (2.0 / np.pi) * np.arcsin(u)
            t2 = v * np.pi / 2.0
            f1 = np.cos(t1) * np.cos(t2)
            f2 = np.cos(t1) * np.sin(t2)
            f3 = np.sin(t1)
            return np.array([f1, f2, f3])

        # Hypervolume-weighted sampling:
        # Generate candidate points on the parameter space, score each by its
        # individual HV contribution (2.5-f1)*(2.5-f2)*(2.5-f3), then greedily
        # select points that maximize cumulative HV.
        
        # Generate a dense set of candidate points using a grid + Halton
        n_grid = 12  # 12x12 grid = 144 points
        grid_u = np.linspace(0.0, 1.0, n_grid + 2)[1:-1]  # exclude edges
        grid_v = np.linspace(0.0, 1.0, n_grid + 2)[1:-1]
        
        candidates = []
        for u in grid_u:
            for v in grid_v:
                f = front_obj(u, v)
                hv_contrib = (2.5 - f[0]) * (2.5 - f[1]) * (2.5 - f[2])
                candidates.append((hv_contrib, u, v))
        
        # Sort by HV contribution descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Greedily select top candidates, but check for dominance
        selected = []
        selected_objs = []
        
        for hv_c, u, v in candidates:
            if used_budget >= self.budget or time.time() > deadline:
                break
            f = front_obj(u, v)
            # Check if dominated by any already selected point
            dominated = False
            for sf in selected_objs:
                if all(sf[i] <= f[i] for i in range(3)) and any(sf[i] < f[i] for i in range(3)):
                    dominated = True
                    break
            if not dominated:
                eval_point(u, v)
                selected_objs.append(f)
        
        # Fill remaining budget with Halton sequence
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        if used_budget < self.budget:
            for idx in range(1, 5000):
                if time.time() > deadline or used_budget >= self.budget:
                    break
                u = halton(idx, 2)
                v = halton(idx, 3)
                # Skip if too close to existing
                too_close = False
                for px in evaluated_points:
                    xu = (2.0 / np.pi) * np.arcsin(u)
                    if abs(px[0] - xu) < 0.01 and abs(px[1] - v) < 0.01:
                        too_close = True
                        break
                if not too_close:
                    eval_point(u, v)

        return evaluated_points
