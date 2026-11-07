"""Optimized optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DFTR-style optimizer with multi-start for ill-conditioned Schaffers F7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _dftr_local(self, func, rng, x0, delta0, max_calls, deadline):
        """
        Derivative-Free Trust-Region: build local quadratic model from 2n+1 points,
        step toward model minimizer within trust region.
        Returns (best_x, best_f, calls_used).
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        calls = 1
        delta = delta0
        no_improve = 0

        while calls < max_calls:
            if time.time() > deadline - 1.5:
                break

            # Build 2n+1 point stencil: center + n forward + n backward
            pts = [x.copy()]
            fvals = [fx]
            for i in range(dim):
                if calls >= max_calls or time.time() > deadline - 1.5:
                    break
                xp = x.copy()
                xp[i] += delta
                xp = np.clip(xp, -5, 5)
                fp = func(xp)
                calls += 1
                pts.append(xp.copy())
                fvals.append(fp)
                if fp < fx:
                    fx = fp
                    x = xp.copy()

                if calls >= max_calls or time.time() > deadline - 1.5:
                    break
                xm = x.copy()
                xm[i] -= delta
                xm = np.clip(xm, -5, 5)
                fm = func(xm)
                calls += 1
                pts.append(xm.copy())
                fvals.append(fm)
                if fm < fx:
                    fx = fm
                    x = xm.copy()

            n_pts = len(pts)
            if n_pts < dim + 1:
                break

            # Estimate gradient via finite differences
            grad = np.zeros(dim)
            for i in range(dim):
                fwd_idx = None
                bwd_idx = None
                for j in range(1, n_pts):
                    dx = pts[j] - x
                    if abs(dx[i]) > 1e-12:
                        if dx[i] > 0:
                            fwd_idx = j
                        else:
                            bwd_idx = j
                if fwd_idx is not None and bwd_idx is not None:
                    h_fwd = pts[fwd_idx][i] - x[i]
                    h_bwd = pts[bwd_idx][i] - x[i]
                    grad[i] = (fvals[fwd_idx] - fvals[bwd_idx]) / (h_fwd - h_bwd)
                elif fwd_idx is not None:
                    h = pts[fwd_idx][i] - x[i]
                    grad[i] = (fvals[fwd_idx] - fvals[0]) / h if h != 0 else 0
                elif bwd_idx is not None:
                    h = pts[bwd_idx][i] - x[i]
                    grad[i] = (fvals[bwd_idx] - fvals[0]) / h if h != 0 else 0

            # Compute diagonal Hessian approximation
            diag_hess = np.zeros(dim)
            for i in range(dim):
                fwd_idx = None
                bwd_idx = None
                for j in range(1, n_pts):
                    dx = pts[j] - x
                    if abs(dx[i]) > 1e-12:
                        if dx[i] > 0:
                            fwd_idx = j
                        else:
                            bwd_idx = j
                if fwd_idx is not None and bwd_idx is not None:
                    h_fwd = pts[fwd_idx][i] - x[i]
                    h_bwd = pts[bwd_idx][i] - x[i]
                    if abs(h_fwd) > 1e-12 and abs(h_bwd) > 1e-12:
                        diag_hess[i] = (fvals[fwd_idx] - 2 * fvals[0] + fvals[bwd_idx]) / (h_fwd * h_bwd)
                if diag_hess[i] <= 0:
                    diag_hess[i] = 1e-6

            # Newton step with diagonal Hessian
            step = -grad / diag_hess
            step_norm = np.linalg.norm(step)

            if step_norm < 1e-10:
                delta *= 0.7
                no_improve += 1
                if no_improve > 5:
                    break
                continue

            # Trust region constraint
            scale = min(1.0, delta / step_norm)
            x_new = x + scale * step
            x_new = np.clip(x_new, -5, 5)

            # Evaluate new point
            if calls >= max_calls or time.time() > deadline - 1.5:
                break
            f_new = func(x_new)
            calls += 1

            if f_new < fx:
                x = x_new
                fx = f_new
                no_improve = 0
                delta = min(10.0, delta * 1.3)
            else:
                delta *= 0.5
                no_improve += 1
                if no_improve > 10:
                    break

            delta = max(1e-6, min(delta, 10.0))

        return x, fx, calls

    def _coordinate_refine(self, func, rng, x0, max_calls, deadline):
        """
        Cyclic coordinate descent with adaptive step sizes for final polishing.
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        calls = 1
        step_sizes = np.full(dim, 0.5)
        no_improve_total = 0

        while calls < max_calls:
            if time.time() > deadline - 1.0:
                break

            improved = False
            for i in range(dim):
                if calls >= max_calls or time.time() > deadline - 1.0:
                    break

                x_try = x.copy()
                x_try[i] += step_sizes[i]
                x_try = np.clip(x_try, -5, 5)
                f_try = func(x_try)
                calls += 1

                if f_try < fx:
                    x = x_try
                    fx = f_try
                    improved = True
                    step_sizes[i] = min(2.0, step_sizes[i] * 1.2)
                else:
                    x_try2 = x.copy()
                    x_try2[i] -= step_sizes[i]
                    x_try2 = np.clip(x_try2, -5, 5)
                    f_try2 = func(x_try2)
                    calls += 1

                    if f_try2 < fx:
                        x = x_try2
                        fx = f_try2
                        improved = True
                        step_sizes[i] = min(2.0, step_sizes[i] * 1.2)
                    else:
                        step_sizes[i] = max(1e-6, step_sizes[i] * 0.7)

            if not improved:
                no_improve_total += 1
                if no_improve_total > 3:
                    break
            else:
                no_improve_total = 0

        return x, fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 300s limit

        dim = self.dim
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        total_calls = 0
        budget = self.budget

        # Phase 1: Multi-start with diverse initial points
        n_starts = 7
        calls_per_start = max(20, budget // (n_starts * 2))

        for i in range(n_starts):
            if total_calls >= budget or time.time() > deadline - 30.0:
                break

            if i == 0:
                x0 = np.zeros(dim)
            elif i == 1:
                x0 = rng.uniform(-1, 1, dim)
            elif i == 2:
                x0 = rng.uniform(-3, 3, dim)
            elif i == 3:
                x0 = rng.uniform(-5, 5, dim)
            else:
                x0 = rng.uniform(-5, 5, dim)

            delta0 = 1.5 if i < 3 else 2.5

            x, fx, calls_used = self._dftr_local(
                func, rng, x0, delta0,
                min(calls_per_start, budget - total_calls),
                deadline
            )
            total_calls += calls_used

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local refinement from best point with DFTR
        if best_x is not None and total_calls < budget:
            remaining = budget - total_calls
            if remaining > 10:
                x, fx, calls_used = self._dftr_local(
                    func, rng, best_x, 0.3,
                    min(remaining // 2, budget - total_calls),
                    deadline
                )
                total_calls += calls_used
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 3: Coordinate descent polishing
        if best_x is not None and total_calls < budget:
            remaining = budget - total_calls
            if remaining > 5:
                x, fx, calls_used = self._coordinate_refine(
                    func, rng, best_x,
                    remaining,
                    deadline
                )
                total_calls += calls_used
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Ensure at least one call was made
        if total_calls == 0:
            x = rng.uniform(-5, 5, dim)
            best_f = func(x)
            best_x = x
            total_calls = 1

        return best_x, best_f
# EVOLVE-BLOCK-END
