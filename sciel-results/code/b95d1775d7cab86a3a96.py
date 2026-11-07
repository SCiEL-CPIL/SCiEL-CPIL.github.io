"""Rewritten optimizer using a local quadratic surrogate model."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Local quadratic surrogate model optimizer for BBOB f12 (Bent cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Initialize with a few random points to build initial history
        rng = np.random.default_rng(42)

        # We'll keep a history of points and function values
        # For the quadratic fit, we need at least n*(n+1)/2 + 1 points
        # But we'll use a sliding window of the last 25 evaluations

        # Start with a reasonable initial point
        # Sample a few points to find a better starting location
        init_points = rng.uniform(-4, 4, (10, n))
        init_vals = [func(p) for p in init_points]
        call_count = 10
        best_idx = np.argmin(init_vals)
        x = init_points[best_idx].copy()
        fx = init_vals[best_idx]
        
        # History: list of (x, f) pairs
        # Include the initial points in history to help the fit
        history = [(p.copy(), v) for p, v in zip(init_points, init_vals)]
        # Ensure current best is in history (it is, as one of the init points)
        # But we want the history to be centered around current x?
        # The fit uses the last 25. Let's just keep the init points.
        
        # Initial step size
        sigma = 1.0

        # Minimum number of points needed for quadratic fit
        min_points = n * (n + 1) // 2 + 1  # 16 for n=5
        # Use up to 25 points for the fit
        max_points = 25
        
        no_improve_count = 0

        while call_count < budget:
            # Check for stagnation and restart
            if no_improve_count > 30:
                # Restart: pick a new random point in the box
                x = rng.uniform(-4, 4, n)
                fx = func(x)
                call_count += 1
                history = [(x.copy(), fx)]
                sigma = 2.0
                no_improve_count = 0
                if call_count >= budget:
                    break

            # If we have enough points, try to fit a quadratic
            if len(history) >= min_points:
                # Take the last max_points evaluations
                recent = history[-max_points:]
                X = np.array([pt for pt, _ in recent])
                F = np.array([f for _, f in recent])

                # Fit quadratic: f(x) ≈ c + x^T b + x^T A x
                # Design matrix for quadratic terms
                # For each point x, create [1, x_1, ..., x_n, x_1^2, x_1*x_2, ..., x_n^2]
                m = n * (n + 1) // 2
                quad_terms = []
                for xi in X:
                    row = [1.0]
                    row.extend(xi)
                    # Squared and cross terms
                    for i in range(n):
                        for j in range(i, n):
                            row.append(xi[i] * xi[j])
                    quad_terms.append(row)
                Q = np.array(quad_terms)

                # Solve least squares
                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(Q, F, rcond=None)
                except np.linalg.LinAlgError:
                    coeffs = None

                if coeffs is not None:
                    c = coeffs[0]
                    b = coeffs[1:1 + n]
                    # Reconstruct A from the quadratic coefficients
                    A = np.zeros((n, n))
                    idx = 1 + n  # start index for quadratic terms
                    for i in range(n):
                        for j in range(i, n):
                            val = coeffs[idx]
                            if i == j:
                                A[i, i] = val
                            else:
                                A[i, j] = val / 2.0
                                A[j, i] = val / 2.0
                            idx += 1

                    # Find minimizer of the quadratic model
                    # grad = b + 2*A*x = 0 => x* = -0.5 * A^{-1} * b
                    try:
                        # Add small regularization for stability
                        # Use a larger regularization to prevent huge steps if A is ill-conditioned
                        reg = 1e-4 * np.trace(A) / n if np.trace(A) > 0 else 1e-4
                        A_reg = A + reg * np.eye(n)
                        x_star = -0.5 * np.linalg.solve(A_reg, b)
                    except np.linalg.LinAlgError:
                        x_star = None

                    if x_star is not None:
                        # Clip to bounds
                        x_star = np.clip(x_star, -5, 5)

                        # Take a step toward x_star, but limit the step size
                        # Use damped step
                        step = x_star - x
                        step_norm = np.linalg.norm(step)
                        
                        # If the step is very small, we might be at the minimum of the model
                        # If the step is large, limit it by sigma
                        if step_norm > sigma:
                            step = step / step_norm * sigma
                        elif step_norm < 1e-5:
                            # Model suggests we are at min, but maybe it's wrong
                            # Fall back to random perturbation
                            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                            fy = func(y)
                            call_count += 1
                            if fy < fx:
                                x, fx = y, fy
                                sigma = min(sigma * 1.2, 4.0)
                                no_improve_count = 0
                            else:
                                sigma = max(sigma * 0.85, 0.05)
                                no_improve_count += 1
                            history.append((y.copy(), fy))
                            continue

                        x_new = np.clip(x + step, -5, 5)
                        fx_new = func(x_new)
                        call_count += 1

                        if fx_new < fx:
                            x, fx = x_new, fx_new
                            # Adapt sigma: increase if we made progress
                            sigma = min(sigma * 1.3, 4.0)
                            no_improve_count = 0
                        else:
                            sigma = max(sigma * 0.7, 0.05)
                            no_improve_count += 1

                        history.append((x_new.copy(), fx_new))
                        continue

            # If quadratic fit failed or we don't have enough points,
            # do a random perturbation
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            call_count += 1

            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.2, 4.0)
                no_improve_count = 0
            else:
                sigma = max(sigma * 0.85, 0.05)
                no_improve_count += 1

            history.append((y.copy(), fy))

        return x, fx
# EVOLVE-BLOCK-END
