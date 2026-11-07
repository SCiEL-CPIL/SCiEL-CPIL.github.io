import numpy as np
import time
import warnings
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        d = self.dim
        lb = np.full(d, -5.0)
        ub = np.full(d, 5.0)
        bounds = [(-5.0, 5.0)] * d

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf

            x = np.asarray(x, dtype=np.float64)
            if x.shape != (d,):
                x = x.reshape(d)
            x = np.clip(x, lb, ub)

            try:
                f = float(func(x))
                if not np.isfinite(f):
                    f = np.inf
            except Exception:
                f = np.inf

            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def remaining_budget():
            return self.budget - calls_made

        def time_ok():
            return time.time() < deadline

        def local_refine(x0, budget_frac=0.5):
            """Run L-BFGS-B then Nelder-Mead from x0, using a fraction of remaining budget."""
            nonlocal best_x, best_f
            if calls_made >= self.budget or not time_ok():
                return

            rem = remaining_budget()
            if rem < 3:
                return

            # L-BFGS-B
            try:
                res = minimize(
                    lambda x: call_func(np.array(x)),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res.x is not None and np.isfinite(res.fun):
                    x0 = res.x
            except Exception:
                pass

            if calls_made >= self.budget or not time_ok():
                return

            # Nelder-Mead
            try:
                res = minimize(
                    lambda x: call_func(np.array(x)),
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if res.x is not None:
                    x0 = res.x
            except Exception:
                pass

        # ---- Phase 1: Initial sampling ----
        # Use a combination of LHS and random to get good coverage
        n_samples = min(60, self.budget // 3)
        for i in range(n_samples):
            if not time_ok() or calls_made >= self.budget:
                break
            if i < n_samples // 2:
                # LHS
                x = np.zeros(d)
                for j in range(d):
                    x[j] = -5.0 + 10.0 * (i + rng.random()) / n_samples
            else:
                x = rng.uniform(-5.0, 5.0, d)
            call_func(x)

        # ---- Phase 2: Iterative refinement with restarts ----
        # Keep a set of good points and periodically restart from them
        good_points = []
        if best_x is not None:
            good_points.append(best_x.copy())

        # Track the best few points
        all_evals = []  # (f, x)

        # Re-evaluate to track all points
        # Actually, let's just track the best points we've seen

        # Main loop: alternate between global exploration and local exploitation
        max_rounds = 20
        for round_idx in range(max_rounds):
            if not time_ok() or calls_made >= self.budget:
                break

            rem = remaining_budget()
            if rem < 10:
                break

            # Determine strategy based on progress
            # Early: more global; Late: more local
            global_frac = max(0.1, 1.0 - round_idx / max_rounds)

            # Global exploration: sample around current best and randomly
            n_global = max(3, int(rem * global_frac * 0.3))
            for _ in range(n_global):
                if not time_ok() or calls_made >= self.budget:
                    break
                if best_x is not None and rng.random() < 0.7:
                    # Sample near best
                    x = best_x + rng.normal(0, 1.0, d)
                else:
                    x = rng.uniform(-5.0, 5.0, d)
                f = call_func(x)
                if np.isfinite(f):
                    all_evals.append((f, x.copy()))

            # Sort and keep top points
            all_evals.sort(key=lambda t: t[0])
            top_k = min(5, len(all_evals))
            for i in range(top_k):
                f_i, x_i = all_evals[i]
                # Check if this is a new good point
                is_new = True
                for gx in good_points:
                    if np.linalg.norm(x_i - gx) < 0.1:
                        is_new = False
                        break
                if is_new:
                    good_points.append(x_i.copy())

            # Local exploitation: refine the best points
            n_refine = min(3, len(good_points))
            for i in range(n_refine):
                if not time_ok() or calls_made >= self.budget:
                    break
                rem = remaining_budget()
                if rem < 5:
                    break
                local_refine(good_points[i])

            # Trim all_evals to avoid memory issues
            if len(all_evals) > 100:
                all_evals = all_evals[:50]

        # ---- Phase 3: Final polish on the global best ----
        if best_x is not None and calls_made < self.budget and time_ok():
            # Multiple passes with decreasing step sizes
            for pass_idx in range(3):
                if not time_ok() or calls_made >= self.budget:
                    break
                rem = remaining_budget()
                if rem < 3:
                    break

                # Tight L-BFGS-B
                try:
                    minimize(
                        lambda x: call_func(np.array(x)),
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-14}
                    )
                except Exception:
                    pass

                if not time_ok() or calls_made >= self.budget:
                    break

                # NM with very tight tolerance
                try:
                    minimize(
                        lambda x: call_func(np.array(x)),
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                    )
                except Exception:
                    pass

        # Fallback
        if best_x is None:
            best_x = np.zeros(d)
            call_func(best_x)

        return best_x, best_f
