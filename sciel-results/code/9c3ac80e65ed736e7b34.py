"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Local quadratic surrogate model optimizer for f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 330s kill

        # Initialize at origin (center of [-5,5]^5)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        call_count = 1

        # History buffer for surrogate fitting
        # Store (x, f) pairs; use last 25 for quadratic fit
        history_x = [x.copy()]
        history_f = [fx]
        max_history = 25

        # Phase 1: Initial exploration to gather diverse points for the surrogate
        # Use a small number of random points to seed the model
        rng = np.random.default_rng(42)
        n_explore = min(10, budget - 1)
        for i in range(n_explore):
            if call_count >= budget or time.time() > deadline:
                break
            # Random point in the box
            y = rng.uniform(-4, 4, dim)
            fy = func(y)
            call_count += 1
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()
            history_x.append(y.copy())
            history_f.append(fy)
            if len(history_x) > max_history:
                history_x.pop(0)
                history_f.pop(0)

        # Phase 2: Surrogate-guided optimization
        # Fit a quadratic model to recent points and step toward predicted minimizer
        n_features = dim * (dim + 1) // 2 + dim + 1  # quadratic + linear + constant

        def build_design_matrix(points):
            """Build design matrix for quadratic model: [1, x_i, x_i*x_j (i<=j)]."""
            n = len(points)
            d = dim
            # Columns: constant, linear (d), quadratic (d*(d+1)/2)
            X = np.ones((n, 1 + d + d * (d + 1) // 2))
            for i in range(n):
                p = points[i]
                col = 1
                # Linear terms
                for j in range(d):
                    X[i, col] = p[j]
                    col += 1
                # Quadratic terms (i <= j)
                for j in range(d):
                    for k in range(j, d):
                        X[i, col] = p[j] * p[k]
                        col += 1
            return X

        def predict_minimizer(points, values):
            """Fit quadratic model and find its unconstrained minimizer."""
            n = len(points)
            if n < dim + 2:
                return None
            X = build_design_matrix(points)
            y = np.array(values)

            # Ridge regularization for stability
            lam = 1e-4
            try:
                A = X.T @ X + lam * np.eye(X.shape[1])
                b = X.T @ y
                coeffs = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                try:
                    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                except:
                    return None

            # Extract Hessian and gradient from quadratic coefficients
            # Model: f(x) = c0 + sum_i c_i x_i + sum_{i<=j} c_ij x_i x_j
            d = dim
            # Gradient: grad_i = c_i + 2*c_ii*x_i + sum_{j!=i} c_ij*x_j
            # Hessian: H_ii = 2*c_ii, H_ij = c_ij + c_ji (but we only have i<=j)
            # For our parameterization: H_ij = c_ij for i!=j, H_ii = 2*c_ii

            # Build full Hessian
            H = np.zeros((d, d))
            grad = np.zeros(d)

            # Linear coefficients start at index 1
            lin_start = 1
            quad_start = 1 + d
            col = quad_start
            for j in range(d):
                for k in range(j, d):
                    c_jk = coeffs[col]
                    if j == k:
                        H[j, j] += 2.0 * c_jk
                        grad[j] += 2.0 * c_jk * points[-1][j]  # approx at center
                    else:
                        H[j, k] += c_jk
                        H[k, j] += c_jk
                        grad[j] += c_jk * points[-1][k]
                        grad[k] += c_jk * points[-1][j]
                    col += 1

            # Add linear terms to gradient
            for i in range(d):
                grad[i] += coeffs[lin_start + i]

            # Minimizer: x* = -H^{-1} * grad
            try:
                H_inv = np.linalg.inv(H)
                x_pred = -H_inv @ grad
            except np.linalg.LinAlgError:
                try:
                    x_pred = -np.linalg.lstsq(H, grad, rcond=None)[0]
                except:
                    return None

            return x_pred

        # Main optimization loop with surrogate guidance
        while call_count < budget and time.time() < deadline:
            # Get recent points for fitting
            n_fit = min(len(history_x), max_history)
            if n_fit < dim + 2:
                # Not enough points, do a small random step
                y = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                history_x.append(y.copy())
                history_f.append(fy)
                if len(history_x) > max_history:
                    history_x.pop(0)
                    history_f.pop(0)
                continue

            # Predict minimizer from surrogate
            recent_x = history_x[-n_fit:]
            recent_f = history_f[-n_fit:]
            x_pred = predict_minimizer(recent_x, recent_f)

            if x_pred is None:
                # Fallback: small random step
                y = np.clip(best_x + 0.3 * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                history_x.append(y.copy())
                history_f.append(fy)
                if len(history_x) > max_history:
                    history_x.pop(0)
                    history_f.pop(0)
                continue

            # Clip predicted point to box
            x_pred = np.clip(x_pred, -5, 5)

            # Take a step toward the predicted minimizer
            # Blend with current best to avoid overshooting
            step_size = 0.7
            y = np.clip(best_x + step_size * (x_pred - best_x), -5, 5)

            # Ensure we're not evaluating the same point
            if np.allclose(y, best_x, atol=1e-10):
                y = np.clip(best_x + 0.1 * rng.standard_normal(dim), -5, 5)

            fy = func(y)
            call_count += 1

            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()

            # Update history
            history_x.append(y.copy())
            history_f.append(fy)
            if len(history_x) > max_history:
                history_x.pop(0)
                history_f.pop(0)

            # Adaptive step: if we're making progress, increase confidence in surrogate
            # If stuck, reduce step and add some exploration
            if call_count % 20 == 0:
                # Check if recent evaluations are improving
                if len(history_f) >= 5:
                    recent_improvement = history_f[-1] < history_f[-5]
                    if not recent_improvement:
                        # Add exploration: evaluate a random nearby point
                        y_exp = np.clip(best_x + 0.3 * rng.standard_normal(dim), -5, 5)
                        fy_exp = func(y_exp)
                        call_count += 1
                        if fy_exp < best_fx:
                            best_fx = fy_exp
                            best_x = y_exp.copy()
                        history_x.append(y_exp.copy())
                        history_f.append(fy_exp)
                        if len(history_x) > max_history:
                            history_x.pop(0)
                            history_f.pop(0)

        return best_x, best_fx
# EVOLVE-BLOCK-END
