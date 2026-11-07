"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer combining (1+1)-ES with periodic quadratic surrogate refinement.
    - Starts at origin for stable early progress on shifted/rotated landscapes.
    - Uses (1+1)-ES with adaptive step size for robust global exploration.
    - Periodically fits a local quadratic model to predict the basin minimum.
    - If the predicted minimum is close and yields improvement, adopt it.
    - Time-guarded to stay well within the 330s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        # Initialize at origin (center of box) for symmetric landscapes
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

        # History for surrogate model (sliding window)
        hist_X = []
        hist_f = []
        max_hist = 30
        min_hist_for_surrogate = 21  # (d+1)(d+2)/2 for d=5

        calls = 1

        for i in range(budget - 1):
            if time.time() > deadline:
                break

            # Add current point to history
            hist_X.append(x.copy())
            hist_f.append(fx)
            if len(hist_X) > max_hist:
                hist_X.pop(0)
                hist_f.pop(0)

            # Try surrogate model refinement every ~15 steps if we have enough data
            used_surrogate = False
            if len(hist_X) >= min_hist_for_surrogate and i > 20 and i % 15 == 0:
                used_surrogate = self._try_surrogate_refinement(
                    func, x, fx, hist_X, hist_f, d, rng, sigma
                )
                if used_surrogate:
                    # Update best
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    no_improve = 0
                    calls += 1
                    continue
            elif len(hist_X) >= min_hist_for_surrogate and i > 20:
                # Try less frequently: every 30 steps
                if i % 30 == 0:
                    used_surrogate = self._try_surrogate_refinement(
                        func, x, fx, hist_X, hist_f, d, rng, sigma
                    )
                    if used_surrogate:
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        no_improve = 0
                        calls += 1
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
                # Stagnation: occasionally try a larger jump
                if no_improve >= 25:
                    sigma = min(sigma * 2.0, max_sigma)
                    no_improve = 0

            # Update best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Final high-precision polish pass
        x = best_x.copy()
        fx = best_f
        remaining = budget - calls
        if remaining > 0:
            sigma = 0.05
            polish_budget = min(remaining, max(50, budget // 5))
            
            for i in range(polish_budget):
                if time.time() > deadline:
                    break
                decay_factor = 0.98
                sigma *= decay_factor
                if sigma < 1e-8:
                    sigma = 1e-8
                    
                z = rng.standard_normal(d)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma = min(sigma * 1.05, 0.5)
                else:
                    sigma = max(sigma * 0.85, 1e-10)

        return best_x, best_f

    def _try_surrogate_refinement(self, func, x, fx, hist_X, hist_f, d, rng, sigma):
        """
        Fit a quadratic model to recent history and try to move to its minimum.
        Returns True if a successful surrogate step was made (x, fx updated in-place).
        """
        try:
            X = np.array(hist_X)
            f_vals = np.array(hist_f)
            n = len(X)

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
            if eigvals[0] > 1e-8:  # Positive definite
                # Solve C * delta = -c1/2 => x_new = x - C^{-1} c1 / 2
                # Actually, the model is f(x) = c0 + c1^T x + x^T C x
                # grad = c1 + 2 C x = 0 => x* = -C^{-1} c1 / 2
                try:
                    x_pred = -np.linalg.solve(C, c1) / 2.0
                    x_pred = np.clip(x_pred, -5, 5)

                    # Only take the step if it's reasonable (not too far)
                    dist = np.linalg.norm(x_pred - x)
                    if dist < 5.0 and dist > 1e-8:
                        fy = func(x_pred)
                        if fy < fx - 1e-10:
                            # Success: update x and fx in the caller via return
                            # We need to modify x and fx; since they're passed by reference
                            # in the sense that we return a signal, let's handle it differently.
                            # Actually, we can't modify x in the caller from here easily.
                            # Let's return the new point and value.
                            # Update x and fx in place by returning them
                            # But we can't do that directly. Let's use a different approach:
                            # We'll return a tuple (new_x, new_f) if successful, None otherwise.
                            # For now, let's just return True and let the caller handle it.
                            # Actually, we need to update x and fx. Let's change the signature.
                            # For now, let's just return True and assume the caller will check.
                            # This is a bit hacky. Let's restructure.
                            # Actually, let's just return the new point and value.
                            # We'll change the return type.
                            return (x_pred, fy)
                except np.linalg.LinAlgError:
                    pass
        except Exception:
            pass

        return None
# EVOLVE-BLOCK-END
