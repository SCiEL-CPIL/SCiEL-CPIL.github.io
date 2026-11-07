"""Model-and-optimize black-box optimizer for BBOB f18 (Schaffers F7).

Uses a Derivative-Free Trust-Region (DFTR) method that constructs local
quadratic surrogates from a fixed set of directional derivatives, leveraging
the high information density of derivative-free approximations to navigate
the ill-conditioned Schaffers F7 landscape more efficiently than
population-based sampling.
"""
import time
import numpy as np


class Optimizer:
    """Derivative-Free Trust-Region (DFTR) optimizer with local quadratic
    surrogate modeling and adaptive trust-region radius."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        WALL = 315.0
        calls = [0]

        def time_ok():
            return (time.time() - t0) < WALL

        def safe(x):
            if calls[0] >= self.budget or not time_ok():
                return None
            try:
                v = float(func(np.asarray(x, dtype=float)))
            except Exception:
                v = 1e18
            if not np.isfinite(v):
                v = 1e18
            calls[0] += 1
            return v

        n = self.dim
        rng = np.random.default_rng(0)

        best_x = np.zeros(n)
        best_f = np.inf

        # DFTR parameters
        delta = 1.0          # trust-region radius
        delta_min = 1e-4
        delta_max = 10.0
        gamma = 2.0          # expansion factor
        rho_min = 0.25       # minimum reduction ratio for acceptance

        # Build initial point
        x = rng.uniform(-3.0, 3.0, n)
        f0 = safe(x)
        if f0 is None:
            return best_x, float(best_f)
        best_f = f0
        best_x = x.copy()

        # Number of directional points for quadratic model: n+1 forward + n+1 backward = 2n+2
        # Plus the center point = 2n+3 total
        # We'll use a simpler approach: evaluate 2n+1 points (n forward, n backward)
        # and build a linear + quadratic model

        while calls[0] < self.budget and time_ok():
            # Build local model around x
            # Evaluate 2n points: x +/- delta*e_i
            points = [x]
            vals = [f0]
            for i in range(n):
                xi_pos = x.copy()
                xi_pos[i] += delta
                xi_pos = np.clip(xi_pos, -5.0, 5.0)
                fi_pos = safe(xi_pos)
                if fi_pos is None:
                    break
                points.append(xi_pos)
                vals.append(fi_pos)

                xi_neg = x.copy()
                xi_neg[i] -= delta
                xi_neg = np.clip(xi_neg, -5.0, 5.0)
                fi_neg = safe(xi_neg)
                if fi_neg is None:
                    break
                points.append(xi_neg)
                vals.append(fi_neg)

            if len(points) < 2 * n + 1:
                # Not enough points, shrink trust region and retry
                delta *= 0.5
                if delta < delta_min:
                    break
                continue

            # Build quadratic model: m(y) = c + g^T y + 0.5 y^T H y
            # where y = x' - x
            # We have values at y=0, y=+delta*e_i, y=-delta*e_i
            c = vals[0]
            g = np.zeros(n)
            H = np.zeros((n, n))

            # First-order differences for gradient
            for i in range(n):
                g[i] = (vals[1 + 2 * i] - vals[2 + 2 * i]) / (2 * delta)

            # Second-order differences for Hessian diagonal
            for i in range(n):
                H[i, i] = (vals[1 + 2 * i] - 2 * vals[0] + vals[2 + 2 * i]) / (delta ** 2)

            # Estimate off-diagonal Hessian entries using cross-derivatives
            # For simplicity, we'll use a sparse Hessian (diagonal only) initially
            # and add off-diagonal terms if budget allows

            # Solve the trust-region subproblem:
            # min c + g^T y + 0.5 y^T H y  s.t. ||y|| <= delta
            # Use a simple Newton step with line search

            # Regularize Hessian
            H_reg = H + 1e-6 * np.eye(n)

            # Try Newton step
            try:
                y_newton = np.linalg.solve(H_reg, -g)
            except np.linalg.LinAlgError:
                y_newton = -g / (np.linalg.norm(g) + 1e-10)

            # Clip to trust region
            norm_y = np.linalg.norm(y_newton)
            if norm_y > delta:
                y_newton = y_newton * (delta / norm_y)

            # Evaluate at the proposed point
            x_trial = x + y_newton
            x_trial = np.clip(x_trial, -5.0, 5.0)
            f_trial = safe(x_trial)
            if f_trial is None:
                break

            # Compute actual reduction
            actual_red = f0 - f_trial

            # Compute predicted reduction
            pred_red = -g @ y_newton - 0.5 * (y_newton @ H_reg @ y_newton)

            # Acceptance criterion
            if pred_red > 1e-10:
                rho = actual_red / pred_red
            else:
                rho = 0.0

            if rho > rho_min:
                # Accept the step
                x = x_trial
                f0 = f_trial
                if f0 < best_f:
                    best_f = f0
                    best_x = x.copy()
                # Expand trust region
                delta = min(delta * gamma, delta_max)
            else:
                # Reject the step, shrink trust region
                delta = max(delta * 0.5, delta_min)

            # If trust region is too small, try a random perturbation
            if delta < delta_min * 1.1:
                # Random restart within a small neighborhood
                x_trial = x + rng.normal(0, 0.1, n)
                x_trial = np.clip(x_trial, -5.0, 5.0)
                f_trial = safe(x_trial)
                if f_trial is None:
                    break
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()
                x = x_trial
                f0 = f_trial
                delta = 0.5

        # Spend remaining budget with (1+1)-ES tail
        if calls[0] < self.budget and time_ok():
            m_tail = best_x.copy()
            sigma_tail = 0.3
            while calls[0] < self.budget and time_ok():
                y = m_tail + (sigma_tail * 0.5) * rng.standard_normal(n)
                y = np.clip(y, -5.0, 5.0)
                fy = safe(y)
                if fy is None:
                    break
                if fy < best_f:
                    best_f, best_x = fy, y.copy()
                    m_tail = best_x.copy()
                    sigma_tail *= 1.15
                else:
                    sigma_tail *= 0.85
                    sigma_tail = float(np.clip(sigma_tail, 1e-8, 10.0))

        return best_x, float(best_f)
