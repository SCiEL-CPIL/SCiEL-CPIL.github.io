"""Rewritten optimizer for shifted/rotated Rosenbrock on [-5, 5]^5."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize as _scipy_minimize


class Optimizer:
    """Global multi-start L-BFGS-B with derivative-free refinement and a
    hard wall-clock deadline guard.  Minimises func on [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.perf_counter()

        # --- deadline guard -------------------------------------------------
        # Evaluator kills the process at 330 s; keep a generous safety margin.
        def deadline():
            return time.perf_counter() - t0 > 250.0

        # --- cheap deterministic multi-start points --------------------
        # We build a set of candidate starting points that cover the box
        # [-5, 5]^dim.  For dim=5 we use a stratified sampling approach.
        rng = np.random.RandomState(42)

        # Stratified points: divide each dimension into 3 intervals
        # and pick a random point in each sub-cell (Latin hypercube style)
        # 3^5 = 243 points, but we'll subsample to stay within budget.
        # Actually let's use a smarter approach:
        # - 1 center point
        # - 2*5 = 10 axis points (±3 along each axis)
        # - Random stratified points

        start_points = [np.zeros(dim)]

        # Axis-aligned points
        for i in range(dim):
            for sign in [-1, 1]:
                pt = np.zeros(dim)
                pt[i] = sign * 3.0
                start_points.append(pt)

        # Stratified random points: 50 points
        n_strat = 50
        for _ in range(n_strat):
            pt = rng.uniform(-4.5, 4.5, size=dim)
            start_points.append(pt)

        # Additional points near the origin (Rosenbrock minimum is often near 0
        # after shifting, but the shift could be anywhere in [-4, 4])
        for _ in range(20):
            pt = rng.uniform(-2.0, 2.0, size=dim)
            start_points.append(pt)

        # Track calls and best
        call_count = 0
        best_x = None
        best_f = float("inf")
        all_evals = []  # list of (f, x) for potential re-use

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return float("inf")
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                return float("inf")
            call_count += 1
            all_evals.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: Evaluate all start points ----------------------------
        for pt in start_points:
            if deadline() or call_count >= budget - 20:
                break
            eval_func(pt)

        # --- Phase 2: Sort by function value and do local refinement -------
        # Sort all evaluated points by function value
        if all_evals:
            all_evals.sort(key=lambda t: t[0])

            # Try local optimization from the top K points
            K = min(10, len(all_evals))
            used_starts = set()

            for idx in range(K):
                if deadline() or call_count >= budget - 5:
                    break
                f_val, x_start = all_evals[idx]

                # Skip if this start is too close to an already-used start
                key = tuple(np.round(x_start, 1))
                if key in used_starts:
                    continue
                used_starts.add(key)

                # L-BFGS-B local optimization
                try:
                    res = _scipy_minimize(
                        eval_func,
                        x_start.copy(),
                        method="L-BFGS-B",
                        bounds=[(-5.0, 5.0)] * dim,
                        options={"maxiter": 80, "maxfun": 60,
                                 "gtol": 1e-10, "ftol": 1e-14},
                    )
                    rx = np.clip(np.asarray(res.x, dtype=float), -5.0, 5.0)
                    eval_func(rx)
                except Exception:
                    pass

                # Nelder-Mead polish if budget remains
                if not deadline() and call_count < budget - 5:
                    try:
                        res2 = _scipy_minimize(
                            eval_func,
                            best_x.copy() if best_x is not None else x_start.copy(),
                            method="Nelder-Mead",
                            bounds=[(-5.0, 5.0)] * dim,
                            options={"maxiter": 100, "maxfev": 40,
                                     "xatol": 1e-10, "fatol": 1e-14},
                        )
                        rx2 = np.clip(np.asarray(res2.x, dtype=float), -5.0, 5.0)
                        eval_func(rx2)
                    except Exception:
                        pass

        # --- Phase 3: Final polish with remaining budget --------------------
        if best_x is not None and not deadline() and call_count < budget - 2:
            try:
                res = _scipy_minimize(
                    eval_func,
                    best_x.copy(),
                    method="Nelder-Mead",
                    bounds=[(-5.0, 5.0)] * dim,
                    options={"maxiter": 200, "maxfev": budget - call_count - 1,
                             "xatol": 1e-12, "fatol": 1e-16},
                )
                rx = np.clip(np.asarray(res.x, dtype=float), -5.0, 5.0)
                eval_func(rx)
            except Exception:
                pass

        # Fallback
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
            except Exception:
                best_f = float("inf")

        return best_x, best_f
# EVOLVE-BLOCK-END
