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

        # Store all evaluations for surrogate/model building
        eval_points = []
        eval_values = []

        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                raise ValueError("Budget exceeded")
            call_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            eval_points.append(x.copy())
            eval_values.append(f)
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
        def local_search(x0, f0, max_calls=150, fine=False, allow_jumps=True):
            nonlocal call_count, best_f, best_x
            x = np.asarray(x0, dtype=float).copy()
            f = float(f0)
            calls_used = 0

            if fine:
                step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
            else:
                step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05]

            # Rastrigin-aware jump targets: near integer values where cos(2*pi*x)=1
            # Global min at 0, local mins at ±1, ±2, etc.
            jump_targets = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.5, -1.5]

            for step in step_sizes:
                if calls_used >= max_calls or call_count >= budget - 5 or not time_ok():
                    break
                improved = True
                while improved and calls_used < max_calls and call_count < budget - 5 and time_ok():
                    improved = False
                    for i in range(dim):
                        if calls_used >= max_calls or call_count >= budget - 5 or not time_ok():
                            break
                        base = x[i]
                        # Standard coordinate steps
                        for delta in [step, -step]:
                            if calls_used >= max_calls or call_count >= budget - 5 or not time_ok():
                                break
                            c = base + delta
                            if not (-4.95 <= c <= 4.95):
                                continue
                            x[i] = c
                            cf = do_call(x)
                            calls_used += 1
                            if cf < f:
                                f = cf
                                improved = True
                            else:
                                x[i] = base
                        # Rastrigin-aware jumps: try to jump toward nearby integer targets
                        if allow_jumps:
                            for target in jump_targets:
                                if calls_used >= max_calls or call_count >= budget - 5 or not time_ok():
                                    break
                                # Only jump if target is close enough to be useful
                                if abs(target - base) < 0.3 or abs(target - base) > 1.5:
                                    continue
                                x[i] = target
                                cf = do_call(x)
                                calls_used += 1
                                if cf < f:
                                    f = cf
                                    improved = True
                                else:
                                    x[i] = base
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return x, f

        # ---------- Quadratic basin-floor model ----------
        def fit_quadratic(x0, f0, n_samples=20):
            """Fit a quadratic model around x0 using nearby evaluations."""
            if len(eval_points) < 10:
                return None
            pts = np.array(eval_points)
            vals = np.array(eval_values)
            # Select points within a reasonable radius of x0
            dists = np.linalg.norm(pts - x0, axis=1)
            mask = dists < 3.0
            if mask.sum() < 8:
                # Use closest points
                idx = np.argsort(dists)[:min(20, len(pts))]
                pts = pts[idx]
                vals = vals[idx]
            else:
                pts = pts[mask]
                vals = vals[mask]
            if len(pts) < dim + 1:
                return None
            # Fit quadratic: f ≈ a + b^T x + x^T C x
            # Use least squares with design matrix
            A = np.ones((len(pts), 1 + dim + dim * (dim + 1) // 2))
            for k in range(len(pts)):
                xk = pts[k]
                A[k, 1:1 + dim] = xk
                col = 1 + dim
                for i in range(dim):
                    for j in range(i, dim):
                        A[k, col] = xk[i] * xk[j]
                        col += 1
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, vals, rcond=None)
            except np.linalg.LinAlgError:
                return None
            # Compute gradient and Hessian at x0
            x0v = np.asarray(x0, dtype=float)
            grad = np.zeros(dim)
            hess = np.zeros((dim, dim))
            for i in range(dim):
                grad[i] = coeffs[1 + i]
                for j in range(i, dim):
                    if i == j:
                        hess[i, j] += 2 * coeffs[1 + dim + i * (dim + 1) // 2 + i * (i - 1) // 2]
                    else:
                        hess[i, j] += coeffs[1 + dim + i * (dim + 1) // 2 + i * (i - 1) // 2]
                        hess[j, i] = hess[i, j]
            # Ensure Hessian is PSD for stable solve
            try:
                eigvals, eigvecs = np.linalg.eigh(hess)
                eigvals = np.maximum(eigvals, 1e-6)
                hess = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except np.linalg.LinAlgError:
                hess = np.eye(dim) * 1e-3
            return grad, hess

        def predict_min_from_model(grad, hess, x0, max_radius=2.0):
            """Predict minimum from quadratic model, clipped to box."""
            try:
                step = np.linalg.solve(hess, -grad)
            except np.linalg.LinAlgError:
                step = -grad / (np.linalg.norm(grad) + 1e-8)
            # Clip step to max_radius
            nrm = np.linalg.norm(step)
            if nrm > max_radius:
                step = step * (max_radius / nrm)
            pred = x0 + step
            pred = np.clip(pred, -4.95, 4.95)
            return pred

        # ---------- Phase 0: Initial exploration ----------
        # Sample points that hit the Rastrigin structure: 0, ±1, ±0.5, random
        init_pts = [np.zeros(dim)]
        for axis in range(dim):
            for s in (-2.0, -1.0, 1.0, 2.0, -0.5, 0.5):
                p = np.zeros(dim)
                p[axis] = s
                init_pts.append(p)
        # Add some random points
        while len(init_pts) < 30:
            init_pts.append(rng.uniform(-3.0, 3.0, dim))

        for p in init_pts:
            if call_count >= budget or not time_ok():
                break
            p = np.clip(p, -4.95, 4.95)
            f = do_call(p)

        # ---------- Phase 1: Basin refinement with model-guided jumps ----------
        reserve = 50
        max_basins = 8
        for basin_iter in range(max_basins):
            if call_count >= budget - reserve or not time_ok():
                break

            # Refine current best with Rastrigin-aware local search
            local_search(best_x, best_f, max_calls=80, fine=False, allow_jumps=True)

            if call_count >= budget - reserve or not time_ok():
                break

            # Build quadratic model and predict next jump target
            model = fit_quadratic(best_x, best_f)
            if model is not None:
                grad, hess = model
                pred = predict_min_from_model(grad, hess, best_x, max_radius=2.5)
                # Blend prediction with a Rastrigin-aware random jump
                jump = rng.normal(0, 0.5, dim)
                seed = np.clip(0.6 * pred + 0.4 * (best_x + jump), -4.95, 4.95)
            else:
                # Fallback: random jump
                n_jump = rng.integers(1, dim + 1)
                jump_coords = rng.choice(dim, size=n_jump, replace=False)
                jump = np.zeros(dim)
                for j in jump_coords:
                    jump[j] = rng.choice([-1.0, 1.0]) * rng.uniform(0.8, 1.5)
                seed = np.clip(best_x + jump, -4.95, 4.95)

            f_seed = do_call(seed)

            # Quick refine from new seed
            if call_count < budget - reserve and time_ok():
                local_search(seed, f_seed, max_calls=40, fine=False, allow_jumps=True)

            # If we found a better point, update best
            if f_seed < best_f:
                best_f = f_seed
                best_x = seed.copy()

        # ---------- Phase 2: Final fine-tune with high-precision polish ----------
        if call_count < budget - 1 and time_ok():
            # Multiple rounds of fine local search
            for _ in range(3):
                if call_count >= budget - 1 or not time_ok():
                    break
                local_search(best_x, best_f, max_calls=60, fine=True, allow_jumps=True)

        # Final polish: very fine steps
        if call_count < budget - 1 and time_ok():
            local_search(best_x, best_f, max_calls=30, fine=True, allow_jumps=False)

        return best_x, best_f
