import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Covariance-free adaptive search with rotation alignment and local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Track calls carefully
        total_calls = [0]

        def tracked_func(x):
            if time.time() >= deadline:
                raise Exception("timeout")
            total_calls[0] += 1
            if total_calls[0] > budget:
                raise Exception("budget exceeded")
            return func(np.clip(x, -5, 5))

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = tracked_func(x)

        best_x = x.copy()
        best_fx = fx

        # Phase 1: Adaptive search with rotation alignment
        # Maintain a rotation matrix R that aligns axes with successful step directions
        R = np.eye(dim)
        sigma = 2.0

        # Reserve last 60 calls for local polish
        polish_budget = 60
        search_budget = budget - polish_budget

        # Track recent successful steps for rotation alignment
        success_steps = []
        max_steps_history = 10

        for _ in range(search_budget - 1):
            if time.time() >= deadline:
                break
            if total_calls[0] >= budget:
                break

            # Generate candidate in rotated space
            z = rng.standard_normal(dim)
            y = x + sigma * R @ z
            y = np.clip(y, -5, 5)
            fy = tracked_func(y)

            if fy < fx:
                # Success: update position and adapt
                step = y - x
                x, fx = y, fy

                # Update rotation matrix to align with successful direction
                # Simple heuristic: rotate the first axis toward the step direction
                step_norm = np.linalg.norm(step)
                if step_norm > 1e-12:
                    step_dir = step / step_norm
                    # Blend current R[0] with step_dir
                    alpha = 0.3  # Adaptation rate
                    new_axis0 = (1 - alpha) * R[0] + alpha * step_dir
                    new_axis0 = new_axis0 / (np.linalg.norm(new_axis0) + 1e-12)
                    R[0] = new_axis0

                    # Re-orthogonalize remaining axes using Gram-Schmidt
                    for i in range(1, dim):
                        R[i] = R[i] - np.dot(R[i], R[0]) * R[0]
                        norm_i = np.linalg.norm(R[i])
                        if norm_i > 1e-12:
                            R[i] = R[i] / norm_i
                        else:
                            # Fallback: use a random orthogonal vector
                            R[i] = rng.standard_normal(dim)
                            R[i] = R[i] - np.dot(R[i], R[0]) * R[0]
                            R[i] = R[i] / (np.linalg.norm(R[i]) + 1e-12)

                    # Keep track of successful steps
                    success_steps.append(step)
                    if len(success_steps) > max_steps_history:
                        success_steps.pop(0)

                # Increase sigma on success
                sigma *= 1.15
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                # Failure: decrease sigma
                sigma *= 0.9

            # Clamp sigma
            if sigma > 10.0:
                sigma = 10.0
            elif sigma < 1e-12:
                sigma = 1e-12

            # Occasionally re-orthogonalize R to prevent drift
            if total_calls[0] % 20 == 0:
                # QR decomposition to maintain orthonormality
                Q, _ = np.linalg.qr(R)
                R = Q

        # Phase 2: Local polish using remaining budget
        remaining = budget - total_calls[0]
        if remaining > 5 and time.time() < deadline:
            nm_calls = [0]

            def nm_func(z):
                if time.time() >= deadline:
                    raise Exception("timeout")
                if nm_calls[0] >= remaining:
                    raise Exception("budget exceeded")
                nm_calls[0] += 1
                return func(np.clip(z, -5, 5))

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
