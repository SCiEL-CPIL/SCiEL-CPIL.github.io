import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def f(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls += 1
            xc = np.clip(x, -5.0, 5.0)
            val = float(func(xc))
            if val < best_f:
                best_f = val
                best_x = xc.copy()
            return val

        # ---- Phase 1: Powell's method (adapts directions to hidden ridge) ----
        # Directions start along the coordinate axes; Powell rotates them to
        # align with the ridge orientation of f13, which simple (1+1)-ES cannot do.
        def powell(x):
            n = dim
            dirs = np.eye(n)
            for _ in range(3):  # a few Powell cycles
                if time.time() > deadline or calls >= budget:
                    break
                old_f = f(x)
                best_delta = 0.0
                best_dir_idx = -1
                last_pt = x.copy()
                for i in range(n):
                    if time.time() > deadline or calls >= budget:
                        break
                    d = dirs[i]
                    # Line search along direction i
                    # Try a few candidate steps to move down
                    base = last_pt
                    fb = f(base)
                    # probe step sizes
                    best_t = 0.0
                    best_val = fb
                    for t in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
                        cand = base + t * d
                        cv = f(cand)
                        if cv < best_val:
                            best_val = cv
                            best_t = t
                    # if best step was 0, try smaller
                    if best_t == 0.0:
                        for t in (-0.25, 0.25):
                            cand = base + t * d
                            cv = f(cand)
                            if cv < best_val:
                                best_val = cv
                                best_t = t
                    new_pt = base + best_t * d
                    delta = fb - best_val
                    if delta > best_delta:
                        best_delta = delta
                        best_dir_idx = i
                    last_pt = new_pt
                # Powell's direction update
                # new point after full cycle
                new_f = f(last_pt)
                if new_f < old_f:
                    # extrapolate in the most productive direction
                    new_dir = last_pt - x
                    # normalize for stability
                    norm = np.linalg.norm(new_dir)
                    if norm > 1e-12:
                        new_dir = new_dir / norm
                    # test the extrapolated direction
                    ext = last_pt + new_dir
                    ext_f = f(ext)
                    if ext_f < new_f:
                        # rotate directions: drop the least productive, add new_dir
                        if best_dir_idx >= 0:
                            dirs = np.delete(dirs, best_dir_idx, axis=0)
                        dirs = np.vstack([dirs, new_dir])
                        x = ext
                    else:
                        x = last_pt
                else:
                    # no improvement; keep x
                    pass
            return x

        # ---- Phase 2: (1+1)-ES refinement with 1/5-style step adaptation ----
        def es(x, fx, iters):
            rng = np.random.default_rng(0)
            sigma = 0.5
            for _ in range(iters):
                if time.time() > deadline or calls >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = f(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
            return x, fx

        # Initialize: a few random starts, keep the best, refine with Powell then ES
        rng0 = np.random.default_rng(1)
        x0 = rng0.uniform(-4.0, 4.0, dim)
        fx0 = f(x0)
        x = x0
        fx = fx0

        # Powell to find a good basin
        x = powell(x)
        fx = f(x)

        # ES to polish
        x, fx = es(x, fx, 120)

        # If budget remains, do another Powell pass from the polished point
        if time.time() < deadline and calls < budget:
            x = powell(x)
            fx = f(x)
            x, fx = es(x, fx, 80)

        return best_x, best_f
