"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Broyden (BFGS-like) derivative-free optimizer with global exploration."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s

        # Phase 1: Global exploration with random restarts
        # Use a set of diverse starting points
        best_x = None
        best_f = np.inf
        best_calls = 0

        # Number of global restarts
        n_restarts = 5
        calls_per_restart = budget // n_restarts

        for r in range(n_restarts):
            if time.time() > deadline - 30:
                break

            # Random start in the box
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            best_calls += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Phase 2: Local Broyden refinement from this start
            # Build inverse Hessian approximation (BFGS update)
            # We'll use a simplified Broyden approach
            # H is the inverse Hessian approximation
            H = np.eye(n)
            sigma = 1.0

            # Broyden iterations
            max_broyden_iters = calls_per_restart - 1
            for it in range(max_broyden_iters):
                if time.time() > deadline - 10:
                    break
                if best_calls >= budget:
                    break

                # Generate a step using current H approximation
                # Use a random direction scaled by H
                direction = H @ rng.standard_normal(n)
                norm_dir = np.linalg.norm(direction)
                if norm_dir < 1e-10:
                    direction = rng.standard_normal(n)
                    norm_dir = np.linalg.norm(direction)

                # Line search: try a few step sizes
                step_size = sigma / norm_dir
                best_local_x = x
                best_local_f = fx
                best_local_step = None

                for alpha in [1.0, 0.5, 0.25, 0.1]:
                    trial = x + alpha * step_size * direction
                    trial = np.clip(trial, -5, 5)
                    f_trial = func(trial)
                    best_calls += 1

                    if f_trial < best_local_f:
                        best_local_f = f_trial
                        best_local_x = trial
                        best_local_step = alpha * step_size

                    if best_calls >= budget:
                        break

                if best_local_step is None or best_calls >= budget:
                    break

                # BFGS update for inverse Hessian
                s = best_local_step * direction  # step taken
                fx_old = fx
                fx_new = best_local_f
                y = (fx_new - fx_old)  # pseudo-gradient difference (scalar)

                # For BFGS with function values only, use a simplified update
                # This is not standard BFGS (which requires gradients), so we
                # use a heuristic update based on the improvement
                if abs(fx_new - fx_old) > 1e-12:
                    # Rank-1 update approximation
                    # H_new = H + (s s^T / (s^T y))  -- simplified
                    s_dot_y = np.dot(s, s) * y  # approximation
                    if abs(s_dot_y) > 1e-12:
                        H = H + np.outer(s, s) / s_dot_y
                        # Ensure H is positive definite
                        H = (H + H.T) / 2

                x = best_local_x
                fx = best_local_f

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                # Adapt sigma based on success
                if fx < fx_old:
                    sigma *= 1.2
                else:
                    sigma *= 0.8

                # Check convergence
                if abs(fx_new - fx_old) < 1e-10:
                    break

        # Phase 3: Final polish with smaller steps
        if best_x is not None and best_calls < budget and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            H = np.eye(n)
            sigma = 0.1

            for _ in range(budget - best_calls):
                if time.time() > deadline - 2:
                    break

                direction = H @ rng.standard_normal(n)
                norm_dir = np.linalg.norm(direction)
                if norm_dir < 1e-10:
                    direction = rng.standard_normal(n)
                    norm_dir = np.linalg.norm(direction)

                step_size = sigma / norm_dir
                trial = x + step_size * direction
                trial = np.clip(trial, -5, 5)
                f_trial = func(trial)
                best_calls += 1

                if f_trial < fx:
                    s = step_size * direction
                    H = H + np.outer(s, s) / (np.dot(s, s) * (f_trial - fx) + 1e-12)
                    H = (H + H.T) / 2
                    x, fx = trial, f_trial
                    sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9

                if sigma < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
