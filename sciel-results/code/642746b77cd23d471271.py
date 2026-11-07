import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """(1+1)-ES with adaptive sigma, covariance adaptation, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Track total calls
        total_calls = 0

        def tracked_func(x):
            nonlocal total_calls
            total_calls += 1
            return func(x)

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = tracked_func(x)

        best_x = x.copy()
        best_fx = fx
        sigma = 2.0

        # Covariance adaptation state
        C = np.eye(dim)  # Covariance matrix
        step_norm = np.zeros(dim)  # Running average of step norms
        success_count = 0
        total_steps = 0

        # Reserve last 50 calls for Nelder-Mead polish
        polish_budget = 50
        es_budget = budget - polish_budget

        for _ in range(es_budget - 1):
            if time.time() >= deadline:
                break
            if total_calls >= budget:
                break

            # Generate candidate using current covariance
            z = rng.standard_normal(dim)
            step = sigma * (C @ z)
            y = np.clip(x + step, -5, 5)
            fy = tracked_func(y)
            total_steps += 1

            # Step norm for adaptation
            step_norm_actual = np.linalg.norm(step)
            step_norm = 0.9 * step_norm + 0.1 * step_norm_actual

            if fy < fx:
                x, fx = y, fy
                success_count += 1
                # Increase sigma on success
                sigma *= 1.3
                # Update covariance: emphasize directions that led to improvement
                step_dir = step / (step_norm_actual + 1e-10)
                C = 0.95 * C + 0.05 * np.outer(step_dir, step_dir)
                # Re-orthogonalize to maintain positive definiteness
                C = 0.5 * (C + C.T)
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                # Decrease sigma on failure
                sigma *= 0.85
                # Slightly shrink covariance in failed direction
                step_dir = step / (step_norm_actual + 1e-10)
                C = 0.98 * C - 0.01 * np.outer(step_dir, step_dir)
                C = 0.5 * (C + C.T)

            # Clamp sigma
            if sigma > 10.0:
                sigma = 10.0
            elif sigma < 1e-12:
                sigma = 1e-12

            # Clamp covariance eigenvalues
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-6:
                C = C + (1e-6 - np.min(eigvals)) * np.eye(dim)

        # Phase 2: Nelder-Mead polish using remaining budget
        remaining = budget - total_calls
        if remaining > 5 and time.time() < deadline:
            nm_calls = [0]

            def nm_func(z):
                nonlocal total_calls
                if time.time() >= deadline:
                    raise Exception("timeout")
                if nm_calls[0] >= remaining:
                    raise Exception("budget exceeded")
                nm_calls[0] += 1
                total_calls += 1
                return func(z)

            # Use Nelder-Mead from best point
            try:
                result = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if result.fun < best_fx:
                    best_x = result.x
                    best_fx = result.fun
            except Exception:
                pass

        return best_x, best_fx
