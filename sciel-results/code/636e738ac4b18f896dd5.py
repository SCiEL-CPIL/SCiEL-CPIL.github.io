"""Global basin search + Rastrigin-aware local refinement for BBOB f4 (Büche-Rastrigin)."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        call_count = 0
        best_x = np.zeros(dim, dtype=float)
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                raise ValueError("Budget exceeded")
            call_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_ok():
            return time.time() < deadline

        # ---------- Rastrigin-aware local search ----------
        # Büche-Rastrigin has cosine ridges; trying multiple offsets per coordinate
        # in a single sweep allows jumping across ridges that small steps cannot.
        def local_search(x0, f0, max_calls=150, fine=False):
            nonlocal call_count, best_f, best_x
            x = np.asarray(x0, dtype=float).copy()
            f = float(f0)
            calls_used = 0

            if fine:
                offsets = [0.1, -0.1, 0.05, -0.05, 0.02, -0.02, 0.01, -0.01, 0.005, -0.005]
            else:
                offsets = [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.25, -0.25, 0.1, -0.1]

            for _ in range(max_calls):
                if calls_used >= max_calls or call_count >= budget - 2 or not time_ok():
                    break
                improved = False
                for i in range(dim):
                    if calls_used >= max_calls or call_count >= budget - 2 or not time_ok():
                        break
                    base = x[i]
                    local_best = None
                    local_best_f = f
                    for off in offsets:
                        if calls_used >= max_calls or call_count >= budget - 2 or not time_ok():
                            break
                        c = base + off
                        if not (-4.95 <= c <= 4.95):
                            continue
                        x[i] = c
                        cf = do_call(x)
                        calls_used += 1
                        if cf < local_best_f:
                            local_best_f = cf
                            local_best = c
                    if local_best is not None:
                        x[i] = local_best
                        f = local_best_f
                        improved = True
                    else:
                        x[i] = base
                if not improved:
                    break

            if f < best_f:
                best_f = f
                best_x = x.copy()
            return x, f

        # ---------- Quadratic model for jump guidance ----------
        cache_x = []
        cache_f = []

        def record(x, f):
            if len(cache_x) < 300:
                cache_x.append(np.asarray(x, dtype=float).copy())
                cache_f.append(float(f))

        def predict_jump():
            if len(cache_x) < dim + 3:
                return None
            pts = np.array(cache_x[:80])
            fs = np.array(cache_f[:80])
            rows = []
            for x in pts:
                row = [1.0] + list(x) + list(x * x)
                rows.append(row)
            A = np.array(rows)
            try:
                coef, *_ = np.linalg.lstsq(A, fs, rcond=None)
            except np.linalg.LinAlgError:
                return None
            a0 = coef[0]
            b = coef[1:1 + dim]
            d = coef[1 + dim:1 + 2 * dim]
            xstar = np.zeros(dim)
            for i in range(dim):
                if abs(d[i]) > 1e-6:
                    xstar[i] = -b[i] / (2.0 * d[i])
                else:
                    xstar[i] = 0.0
            xstar = np.clip(xstar, -4.5, 4.5)
            return xstar

        # ---------- Phase 0: Initial exploration ----------
        init_pts = [np.zeros(dim)]
        for axis in range(dim):
            for s in (-1.5, 1.5, -2.5, 2.5):
                p = np.zeros(dim)
                p[axis] = s
                init_pts.append(p)
        while len(init_pts) < 24:
            init_pts.append(rng.uniform(-4.0, 4.0, dim))

        for p in init_pts:
            if call_count >= budget or not time_ok():
                break
            p = np.clip(p, -4.95, 4.95)
            f = do_call(p)
            record(p, f)

        # ---------- Phase 1: Multi-basin refinement loop ----------
        reserve = 50
        max_basins = 8
        for b in range(max_basins):
            if call_count >= budget - reserve or not time_ok():
                break

            # Refine current best basin
            x, f = local_search(best_x, best_f, max_calls=100, fine=False)
            record(x, f)

            if call_count >= budget - reserve or not time_ok():
                break

            # Determine jump target
            pred = predict_jump()
            if pred is not None and rng.random() < 0.6:
                seed = pred + rng.normal(0, 0.4, dim)
            else:
                # Random jump from best
                n_jump = rng.integers(1, dim + 1)
                jump_coords = rng.choice(dim, size=n_jump, replace=False)
                jump = np.zeros(dim)
                for j in jump_coords:
                    jump[j] = rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 2.5)
                seed = best_x + jump
            seed = np.clip(seed, -4.95, 4.95)
            f_seed = do_call(seed)
            record(seed, f_seed)

            # Quick refine from new seed
            if call_count < budget - reserve and time_ok():
                x2, f2 = local_search(seed, f_seed, max_calls=60, fine=False)
                record(x2, f2)

        # ---------- Phase 2: Final fine-tune ----------
        if call_count < budget - 1 and time_ok():
            local_search(best_x, best_f, max_calls=200, fine=True)

        return best_x, best_f
