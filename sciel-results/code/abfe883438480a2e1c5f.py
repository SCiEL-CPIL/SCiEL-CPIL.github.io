"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with quadratic surrogate refinement for anisotropic landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        # Initialize at origin
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        sigma = 2.5
        min_sigma = 1e-6
        max_sigma = 10.0
        growth = 1.15
        decay = 0.92
        no_improve = 0
        calls = 1

        # History for surrogate model
        hist_X = [x.copy()]
        hist_f = [fx]
        max_hist = 40
        min_hist_for_surrogate = 21  # (d+1)(d+2)/2 for d=5

        for i in range(budget - 1):
            if time.time() > deadline:
                break

            # Try surrogate refinement periodically
            if len(hist_X) >= min_hist_for_surrogate and i > 10 and i % 20 == 0:
                new_x, new_f = self._try_surrogate(func, x, fx, hist_X, hist_f, d)
                if new_x is not None:
                    x, fx = new_x, new_f
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    calls += 1
                    no_improve = 0
                    # Add to history
                    hist_X.append(x.copy())
                    hist_f.append(fx)
                    if len(hist_X) > max_hist:
                        hist_X.pop(0)
                        hist_f.pop(0)
                    continue

            # Standard (1+1)-ES step
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * growth, max_sigma)
                no_improve = 0
            else:
                sigma = max(sigma * decay, min_sigma)
                no_improve += 1
                if no_improve >= 25:
                    sigma = min(sigma * 2.0, max_sigma)
                    no_improve = 0

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Update history
            hist_X.append(x.copy())
            hist_f.append(fx)
            if len(hist_X) > max_hist:
                hist_X.pop(0)
                hist_f.pop(0)

        # Final polish phase
        remaining = budget - calls
        if remaining > 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            polish_steps = min(remaining, 80)

            for _ in range(polish_steps):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.1, 0.5)
                else:
                    sigma = max(sigma * 0.85, 1e-10)

        return best_x, best_f

    def _try_surrogate(self, func, x, fx, hist_X, hist_f, d):
        """
        Fit a quadratic model to recent history and return (new_x, new_f)
        if the predicted minimum is valid and better, else (None, None).
        """
        try:
            X = np.array(hist_X)
            f_vals = np.array(hist_f)
            n = len(X)
            if n < 21:
                return None, None

            # Build design matrix for quadratic: [1, x, x_i*x_j for i<=j]
            d_quad = d * (d + 1) // 2
            A = np.ones((n, 1 + d + d_quad))

            for k in range(n):
                A[k, 1:1 + d] = X[k]
                idx = 1 + d
                for ii in range(d):
                    for jj in range(ii, d):
                        A[k, idx] = X[k, ii] * X[k, jj]
                        idx += 1

            # Solve least squares
            coeffs, _, _, _ = np.linalg.lstsq(A, f_vals, rcond=None)

            # Extract linear and quadratic coefficients
            c1 = coeffs[1:1 + d]

            # Reconstruct symmetric matrix C from quadratic coefficients
            C = np.zeros((d, d))
            idx = 1 + d
            for ii in range(d):
                for jj in range(ii, d):
                    val = coeffs[idx]
                    C[ii, jj] = val
                    if ii != jj:
                        C[jj, ii] = val
                    idx += 1

            # Check if Hessian (2C) is positive definite -> local minimum
            eigvals = np.linalg.eigvalsh(C)
            if eigvals[0] > 1e-6:  # Positive definite
                # Solve C * delta = -c1/2 => x_new = -C^{-1} c1 / 2
                try:
                    x_pred = -np.linalg.solve(C, c1) / 2.0
                    x_pred = np.clip(x_pred, -5, 5)

                    # Only take the step if it's reasonable
                    dist = np.linalg.norm(x_pred - x)
                    if dist < 8.0 and dist > 1e-8:
                        fy = func(x_pred)
                        if fy < fx - 1e-10:
                            return x_pred, fy
                except np.linalg.LinAlgError:
                    pass
        except Exception:
            pass

        return None, None
# EVOLVE-BLOCK-END
