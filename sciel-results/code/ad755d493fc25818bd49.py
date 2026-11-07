"""Rewritten optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start adaptive coordinate search for ill-conditioned functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Track global best
        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Deterministic quasi-random initial points using Halton-like sequence
        # We'll use a simple low-discrepancy approach
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def quasi_random_points(n, dim):
            points = []
            for i in range(1, n + 1):
                p = np.array([halton(i, b) for b in range(2, dim + 2)])
                # Map to [-5, 5]
                p = -5.0 + 10.0 * p
                points.append(p)
            return points

        # Phase 1: Initial sampling with multiple starts
        n_starts = min(10, budget // 50)  # At least 10 starts if budget allows
        initial_points = quasi_random_points(n_starts, d)

        for p in initial_points:
            if not time_ok() or calls_made >= budget:
                break
            fx = safe_call(p)
            if fx is None:
                break

        # Phase 2: Adaptive coordinate search from best point
        if best_x is not None:
            x = best_x.copy()
            fx = best_f

            # Per-dimension step sizes, initialized adaptively
            steps = np.full(d, 1.0)

            # Try to find good initial step sizes per dimension
            for dim_i in range(d):
                if not time_ok() or calls_made >= budget:
                    break
                # Try a few step sizes to find one that works
                for trial in range(3):
                    trial_step = steps[dim_i] * (0.5 ** trial)
                    if trial_step < 1e-4:
                        break
                    y = x.copy()
                    y[dim_i] += trial_step
                    fy = safe_call(y)
                    if fy is None:
                        break
                    if fy < fx:
                        x = y
                        fx = fy
                        break
                    y[dim_i] -= 2 * trial_step
                    fy = safe_call(y)
                    if fy is None:
                        break
                    if fy < fx:
                        x = y
                        fx = fy
                        break
                    steps[dim_i] = trial_step * 0.5

            # Main coordinate search loop with momentum
            momentum = np.zeros(d)
            learning_rate = 0.7

            while time_ok() and calls_made < budget:
                # Coordinate descent with adaptive steps
                improved = False
                for dim_i in range(d):
                    if not time_ok() or calls_made >= budget:
                        break

                    # Try positive step
                    y = x.copy()
                    y[dim_i] += steps[dim_i]
                    fy = safe_call(y)
                    if fy is None:
                        break

                    if fy < fx:
                        x = y
                        fx = fy
                        momentum[dim_i] = learning_rate * momentum[dim_i] + (1 - learning_rate) * steps[dim_i]
                        steps[dim_i] = max(1e-5, steps[dim_i] * 1.2)
                        improved = True
                    else:
                        # Try negative step
                        y = x.copy()
                        y[dim_i] -= steps[dim_i]
                        fy = safe_call(y)
                        if fy is None:
                            break

                        if fy < fx:
                            x = y
                            fx = fy
                            momentum[dim_i] = learning_rate * momentum[dim_i] - (1 - learning_rate) * steps[dim_i]
                            steps[dim_i] = max(1e-5, steps[dim_i] * 1.2)
                            improved = True
                        else:
                            # No improvement, reduce step
                            steps[dim_i] *= 0.7
                            momentum[dim_i] *= 0.5

                    # Update best
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                if not improved and calls_made < budget:
                    # Try a random perturbation to escape local minima
                    if time_ok():
                        perturbation = np.random.default_rng(int(time.time() * 1000) % 10000).standard_normal(d) * 0.5
                        y = x + perturbation
                        fy = safe_call(y)
                        if fy is not None and fy < fx:
                            x = y
                            fx = fy
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            # Reset some steps
                            steps *= 1.5
                            momentum *= 0.5
                        else:
                            # Shrink steps further
                            steps *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
