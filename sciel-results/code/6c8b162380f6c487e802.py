"""Rewritten optimizer for BBOB f12 (Bent cigar) with quadratic surrogate + adaptive ES."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # --- Phase 1: Initial exploration with adaptive (1+1)-ES ---
        # Start with a few random points to get a feel for the landscape
        best_x = rng.uniform(-5, 5, dim)
        best_f = func(best_x)
        calls = 1

        # Keep a history of evaluations for the quadratic model
        history_x = [best_x.copy()]
        history_f = [best_f]

        # Adaptive step size
        sigma = 3.0
        success_count = 0
        trail = 0.0
        success_ratio = 0.2  # target 1/5 rule

        # Exploration phase: use (1+1)-ES for first portion of budget
        explore_budget = int(budget * 0.4)
        for i in range(explore_budget):
            if calls >= budget or time.time() > deadline:
                break

            # Generate trial point
            y = best_x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            history_x.append(y.copy())
            history_f.append(fy)

            if fy < best_f:
                best_x = y.copy()
                best_f = fy
                success_count += 1
                sigma *= 1.25
                sigma = min(sigma, 15.0)
            else:
                sigma *= 0.87
                sigma = max(sigma, 0.01)

            # Keep only last 50 points for model
            if len(history_x) > 50:
                history_x = history_x[-50:]
                history_f = history_f[-50:]

        # --- Phase 2: Quadratic surrogate model refinement ---
        # Fit a quadratic model to recent evaluations and use it to guide steps
        polish_budget = budget - calls
        for i in range(polish_budget):
            if calls >= budget or time.time() > deadline:
                break

            # Build quadratic model from recent history
            # Use last min(25, len) points
            n_hist = min(25, len(history_x))
            if n_hist >= dim + 1:  # need at least dim+1 points
                X = np.array(history_x[-n_hist:])
                F = np.array(history_f[-n_hist:])

                # Fit quadratic: f(x) ≈ c + b^T x + x^T A x
                # Build design matrix for quadratic features
                d = dim
                # Number of quadratic terms: 1 + d + d*(d+1)/2
                n_params = 1 + d + d * (d + 1) // 2

                # Build feature matrix
                M = np.ones((n_hist, n_params))
                # Linear terms: columns 1..d
                M[:, 1:d + 1] = X
                # Quadratic terms: columns d+1 onwards
                col = d + 1
                for i_idx in range(d):
                    for j_idx in range(i_idx, d):
                        M[:, col] = X[:, i_idx] * X[:, j_idx]
                        col += 1

                # Solve least squares
                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(M, F, rcond=None)
                    if rank >= n_params - 1:  # well-conditioned
                        # Extract Hessian from quadratic coefficients
                        # Quadratic part: sum_{i<=j} coeffs[d+1+idx(i,j)] * x_i * x_j
                        # Hessian H: H[i,j] = coeffs[d+1+idx(i,j)] for i!=j, H[i,i] = 2*coeffs[d+1+idx(i,i)]
                        H = np.zeros((d, d))
                        col = d + 1
                        for i_idx in range(d):
                            for j_idx in range(i_idx, d):
                                H[i_idx, j_idx] = coeffs[col]
                                if i_idx != j_idx:
                                    H[j_idx, i_idx] = coeffs[col]
                                else:
                                    H[i_idx, i_idx] = 2 * coeffs[col]
                                col += 1

                        # Gradient at best_x
                        grad = coeffs[1:d + 1].copy()
                        for i_idx in range(d):
                            for j_idx in range(d):
                                if i_idx != j_idx:
                                    grad[i_idx] += H[i_idx, j_idx] * best_x[j_idx]
                            grad[i_idx] += H[i_idx, i_idx] * best_x[i_idx]

                        # Check if Hessian is positive definite enough
                        eigvals = np.linalg.eigvalsh(H)
                        if np.min(eigvals) > 1e-8:
                            # Newton step: x_new = x - H^{-1} grad
                            try:
                                step = np.linalg.solve(H, grad)
                                # Damp the step to avoid overshooting
                                step_norm = np.linalg.norm(step)
                                if step_norm > sigma:
                                    step = step * sigma / step_norm
                                y = best_x - step
                                y = np.clip(y, -5, 5)
                            except np.linalg.LinAlgError:
                                y = best_x + sigma * rng.standard_normal(dim)
                                y = np.clip(y, -5, 5)
                        else:
                            # Fallback: use gradient descent step
                            grad_norm = np.linalg.norm(grad)
                            if grad_norm > 1e-12:
                                y = best_x - 0.5 * grad / grad_norm * sigma
                                y = np.clip(y, -5, 5)
                            else:
                                y = best_x + sigma * rng.standard_normal(dim)
                                y = np.clip(y, -5, 5)
                    else:
                        y = best_x + sigma * rng.standard_normal(dim)
                        y = np.clip(y, -5, 5)
                except Exception:
                    y = best_x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
            else:
                y = best_x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)

            fy = func(y)
            calls += 1

            history_x.append(y.copy())
            history_f.append(fy)

            if fy < best_f:
                best_x = y.copy()
                best_f = fy
                # Reduce sigma on success in polish phase
                sigma *= 0.8
                sigma = max(sigma, 0.001)
            else:
                sigma *= 0.9
                sigma = max(sigma, 0.001)

            # Keep only last 50 points
            if len(history_x) > 50:
                history_x = history_x[-50:]
                history_f = history_f[-50:]

        return best_x, best_f
