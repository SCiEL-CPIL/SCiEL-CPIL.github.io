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
        deadline = start_time + 285  # safety margin before 300s limit

        call_count = 0
        best_x = np.zeros(dim, dtype=float)
        best_f = float('inf')

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

        def remaining():
            return budget - call_count

        # ---------- Rastrigin-aware local search ----------
        # Büche-Rastrigin = sum_i [10*(x_i^2 - 1) + cos(2*pi*x_i)] + 10*dim.
        # The cosine term creates ridges; a coordinate move that can jump across
        # a ridge (not just step) is the key to escaping local minima.
        def local_search(x0, f0, max_calls=150):
            nonlocal call_count, best_f, best_x
            x = np.asarray(x0, dtype=float).copy()
            f = float(f0)
            iters = 0
            for _ in range(max_calls):
                if call_count >= budget - 2 or not time_ok():
                    break
                iters += 1
                improved = False
                for i in range(dim):
                    if call_count >= budget - 2 or not time_ok():
                        break
                    base = x[i]
                    # candidate positions to try along this coordinate
                    cands = [
                        base + 0.5, base - 0.5,
                        base + 1.0, base - 1.0,
                        base + 2.0, base - 2.0,
                        base + 0.25, base - 0.25,
                        base + 0.1, base - 0.1,
                    ]
                    local_best = None
                    local_best_f = f
                    for c in cands:
                        if call_count >= budget - 2 or not time_ok():
                            break
                        if not (-4.9 <= c <= 4.9):
                            continue
                        x[i] = c
                        cf = do_call(x)
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
            return x, f, iters

        # ---------- Quadratic model of the basin floor ----------
        # Fit f(x) ~ a0 + b.x + x^T D x  (diagonal quadratic, cheap).
        # Use it to (a) predict the basin minimum and (b) guide the next global seed.
        def fit_quadratic():
            # gather a set of evaluated points with distinct coordinates
            # We rebuild from a small cache of (x, f) we keep below.
            pass

        # ---------- Global seeding + refinement loop ----------
        # Keep a cache of evaluated points to fit a model.
        cache_x = []
        cache_f = []

        def record(x, f):
            if len(cache_x) < 400:
                cache_x.append(np.asarray(x, dtype=float).copy())
                cache_f.append(float(f))

        def predict_min_from_model():
            # Fit diagonal quadratic on the best 40 cache points.
            pts = []
            fs = []
            for x, f in zip(cache_x, cache_f):
                pts.append(x)
                fs.append(f)
            if len(pts) < dim + 2:
                return None
            pts = np.array(pts[:60])
            fs = np.array(fs[:60])
            # design matrix: [1, x, x_i^2]
            rows = []
            for x in pts:
                row = [1.0] + list(x) + list(x * x)
                rows.append(row)
            A = np.array(rows)
            y = fs
            try:
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            except np.linalg.LinAlgError:
                return None
            a0 = coef[0]
            b = coef[1:1 + dim]
            d = coef[1 + dim:1 + 2 * dim]
            # solve (2D + b) = 0 -> x* = -b / (2 d) where d != 0
            xstar = np.zeros(dim)
            for i in range(dim):
                if abs(d[i]) > 1e-6:
                    xstar[i] = -b[i] / (2.0 * d[i])
                else:
                    xstar[i] = 0.0
            xstar = np.clip(xstar, -4.5, 4.5)
            return xstar

        # Phase 0: initial global exploration (stratified grid + random)
        n_init = 20
        init_pts = [np.zeros(dim)]
        for axis in range(dim):
            for s in (-1.5, 1.5):
                p = np.zeros(dim)
                p[axis] = s
                init_pts.append(p)
        rng = np.random.default_rng(1234)
        while len(init_pts) < n_init:
            init_pts.append(rng.uniform(-4.0, 4.0, dim))

        for p in init_pts:
            if call_count >= budget or not time_ok():
                break
            p = np.clip(p, -4.9, 4.9)
            f = do_call(p)
            record(p, f)

        # Phase 1: iterate {refine best basin, use model to jump to next basin}
        # Budget split: reserve a final fine-tune; spend the rest here.
        reserve = 30
        while call_count < budget - reserve and time_ok():
            # refine the current best basin with the local search
            x, f, iters = local_search(best_x, best_f, max_calls=120)
            record(x, f)

            # if we made no progress, use the model / random to jump
            pred = predict_min_from_model()
            if pred is not None:
                seed = pred + rng.normal(0, 0.3, dim)
            else:
                seed = rng.uniform(-4.0, 4.0, dim)
            seed = np.clip(seed, -4.9, 4.9)
            f_seed = do_call(seed)
            record(seed, f_seed)
            # quick local refine from the new seed
            x2, f2, _ = local_search(seed, f_seed, max_calls=80)
            record(x2, f2)

        # Phase 2: final fine-tune from the overall best with a tight local search
        if call_count < budget - 1 and time_ok():
            local_search(best_x, best_f, max_calls=200)

        return best_x, best_f
