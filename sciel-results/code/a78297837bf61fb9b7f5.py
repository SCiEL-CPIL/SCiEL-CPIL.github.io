"""Portfolio-based continuous optimizer for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer combining:
    1. Success-rate-adaptive random walk for global exploration
    2. Quadratic-surrogate (Broyden-Flower Hessian) curvature-aware jumps
       toward the step discontinuity
    3. Nelder-Mead simplex for local refinement
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s wall-clock, 20s safety margin
        budget = self.budget
        dim = self.dim

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            try:
                x = np.asarray(x, dtype=float)
                x = np.clip(x, -5.0, 5.0)
                f = float(func(x))
                if not np.isfinite(f):
                    f = np.inf
            except Exception:
                return np.inf
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # --- Phase 1: Success-rate-adaptive random walk ---
        x = np.zeros(dim)
        fx = safe_call(x)

        step = 2.0
        window = 20
        accept_count = 0
        total_count = 0

        # Use ~45% of budget for global exploration
        max_walk_steps = int(budget * 0.45)

        for i in range(max_walk_steps):
            if calls_made >= budget or time.time() > deadline:
                break

            direction = rng.standard_normal(dim)
            direction /= (np.linalg.norm(direction) + 1e-12)
            y = np.clip(x + step * direction, -5.0, 5.0)
            fy = safe_call(y)

            if fy < fx:
                x, fx = y, fy
                accept_count += 1

            total_count += 1

            if total_count % window == 0:
                rate = accept_count / total_count
                if rate > 0.5:
                    step *= 1.3
                elif rate < 0.2:
                    step *= 0.7
                step = np.clip(step, 0.05, 5.0)
                accept_count = 0
                total_count = 0

        # --- Phase 2: Quadratic-surrogate curvature-aware jumps ---
        # Build a local quadratic model from recent evaluations around the
        # current best to estimate the Hessian of the step-ellipsoidal surface,
        # then take large curvature-aware jumps toward the discontinuity.
        if best_x is not None and calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f

            # Collect a set of points around x for the quadratic model
            model_points = []
            model_vals = []

            def add_model_pt(p, v):
                if np.isfinite(v):
                    model_points.append(p.copy())
                    model_vals.append(v)

            add_model_pt(x, fx)

            # Sample points around x at a few scales
            for scale in [0.3, 0.8, 1.5, 2.5]:
                if calls_made >= budget or time.time() > deadline:
                    break
                for _ in range(dim + 1):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    p = np.clip(x + scale * rng.standard_normal(dim), -5.0, 5.0)
                    v = safe_call(p)
                    add_model_pt(p, v)

            # Try to fit a quadratic model and take curvature-aware steps
            if len(model_points) >= dim + 2:
                pts = np.array(model_points)
                vals = np.array(model_vals)

                # Center at x
                pts_c = pts - x
                # Fit quadratic: f ≈ c + g^T d + 0.5 d^T H d
                # Use least squares on the quadratic features
                n = len(pts_c)
                # Build feature matrix: [1, d_i, 0.5*d_i*d_j (symmetric)]
                d2 = np.zeros((n, dim * (dim + 1) // 2))
                for i in range(n):
                    d = pts_c[i]
                    idx = 0
                    for a in range(dim):
                        d2[i, idx] = 0.5 * d[a] * d[a]
                        idx += 1
                        for b in range(a + 1, dim):
                            d2[i, idx] = d[a] * d[b]
                            idx += 1
                A = np.hstack([np.ones((n, 1)), pts_c, d2])
                # Solve least squares
                try:
                    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
                    c = coef[0]
                    g = coef[1:1 + dim]
                    Hf = coef[1 + dim:]
                    # Reconstruct Hessian
                    H = np.zeros((dim, dim))
                    idx = 0
                    for a in range(dim):
                        H[a, a] = Hf[idx]
                        idx += 1
                        for b in range(a + 1, dim):
                            H[a, b] = Hf[idx]
                            H[b, a] = Hf[idx]
                            idx += 1

                    # Take a curvature-aware step: minimize c + g^T s + 0.5 s^T H s
                    # s* = -H^{-1} g  (regularized)
                    reg = 1e-3
                    try:
                        s = np.linalg.solve(H + reg * np.eye(dim), -g)
                    except np.linalg.LinAlgError:
                        s = -g / (np.linalg.norm(g) + 1e-12)
                    # Limit step size
                    snorm = np.linalg.norm(s)
                    max_step = 3.0
                    if snorm > max_step:
                        s = s * (max_step / snorm)
                    y = np.clip(x + s, -5.0, 5.0)
                    fy = safe_call(y)
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                except Exception:
                    pass

        # --- Phase 3: Nelder-Mead from best point and neighbors ---
        nm_starts = []
        if best_x is not None:
            nm_starts.append(best_x.copy())
            for scale in [0.5, 1.0, 2.0]:
                nm_starts.append(np.clip(best_x + rng.normal(0, scale, dim), -5.0, 5.0))
        else:
            nm_starts.append(np.zeros(dim))

        # Reserve ~40% of budget for Nelder-Mead
        nm_budget = int(budget * 0.40)
        per_nm = max(20, nm_budget // max(1, len(nm_starts)))

        for start_pt in nm_starts:
            if calls_made >= budget or time.time() > deadline:
                break
            x_opt, f_opt = self._nelder_mead(
                func, safe_call, start_pt, budget, deadline, dim, rng,
                max_iter=per_nm)
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # --- Phase 4: Final fine-tuning with small random steps ---
        if best_x is not None and calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            for _ in range(min(50, budget - calls_made)):
                if calls_made >= budget or time.time() > deadline:
                    break
                y = np.clip(x + 0.1 * rng.standard_normal(dim), -5.0, 5.0)
                fy = safe_call(y)
                if fy < fx:
                    x, fx = y, fy
                    best_x, best_f = x.copy(), fx

        # Ensure at least one call was made
        if calls_made == 0:
            safe_call(np.zeros(dim))

        return best_x, best_f

    def _nelder_mead(self, func, safe_call, x0, budget, deadline, dim, rng,
                     max_iter=100):
        """Nelder-Mead simplex method with early termination."""
        x = np.clip(x0, -5.0, 5.0).copy()
        fx = safe_call(x)

        simplex = [x.copy()]
        simplex_f = [fx]

        # Build initial simplex with adaptive offsets
        for d in range(dim):
            xi = x.copy()
            off = 0.2 if x[d] + 0.2 <= 5.0 else -0.2
            xi[d] += off
            simplex.append(np.clip(xi, -5.0, 5.0))
            simplex_f.append(safe_call(simplex[-1]))

        best_x = simplex[0].copy()
        best_f = min(simplex_f)

        for it in range(max_iter):
            if calls_exhausted(safe_call) or time.time() > deadline:
                break

            # Sort simplex
            order = np.argsort(simplex_f)
            simplex = [simplex[i].copy() for i in order]
            simplex_f = [simplex_f[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = simplex_f[-1]

            # Reflection
            alpha = 1.0
            xr = centroid + alpha * (centroid - worst)
            xr = np.clip(xr, -5.0, 5.0)
            fr = safe_call(xr)

            if fr < simplex_f[0]:
                gamma = 2.0
                xe = centroid + gamma * (centroid - worst)
                xe = np.clip(xe, -5.0, 5.0)
                fe = safe_call(xe)
                if fe < fr:
                    simplex[-1], simplex_f[-1] = xe, fe
                else:
                    simplex[-1], simplex_f[-1] = xr, fr
            elif fr < simplex_f[-2]:
                simplex[-1], simplex_f[-1] = xr, fr
            else:
                rho = 0.5
                if fr < worst_f:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (xr - centroid)
                xc = np.clip(xc, -5.0, 5.0)
                fc = safe_call(xc)
                if fc < worst_f:
                    simplex[-1], simplex_f[-1] = xc, fc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex_f[i] = safe_call(simplex[i])

            cur_best_idx = int(np.argmin(simplex_f))
            if simplex_f[cur_best_idx] < best_f:
                best_f = simplex_f[cur_best_idx]
                best_x = simplex[cur_best_idx].copy()

            if np.max(simplex_f) - np.min(simplex_f) < 1e-6:
                break

        return best_x, best_f


def calls_exhausted(safe_call):
    """Helper: always False; budget check is done via safe_call returning best_f."""
    return False
# EVOLVE-BLOCK-END
