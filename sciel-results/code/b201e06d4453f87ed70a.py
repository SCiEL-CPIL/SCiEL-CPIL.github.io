"""
Strategy: Multi-Start Powell's Method with adaptive restarts.
Rationale: The Discus function (f11) is highly anisotropic with a flat plateau.
Powell's method uses a set of conjugate directions to minimize along, dynamically
replacing the worst direction with a new one formed by the overall step. This
efficiently handles the anisotropic curvature without requiring gradient
approximations or covariance matrix updates, allowing for rapid convergence from
multiple diverse starting points.
"""

import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s kill

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        # Helper to safely call func and track budget
        def safe_call(x):
            nonlocal total_calls, best_f, best_x
            if total_calls >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            total_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # --- Powell's Method with adaptive step sizes ---
        def powell_method(x0, f0, max_calls, initial_step=1.0):
            """
            Powell's method with adaptive step sizes and convergence checks.
            """
            nonlocal total_calls, best_f, best_x

            if total_calls >= budget:
                return best_x, best_f

            start_calls = total_calls

            # Initialize directions as standard basis
            directions = np.eye(dim)

            # Initialize step sizes
            step_sizes = np.full(dim, initial_step)

            # Current point and function value
            x_curr = x0.copy()
            f_curr = f0

            # Track best in this Powell run
            best_x_powell = x_curr.copy()
            best_f_powell = f_curr

            # Number of consecutive iterations without improvement
            no_improve = 0
            prev_best = best_f

            while total_calls - start_calls < max_calls and total_calls < budget:
                if time.time() > deadline - 5:
                    break

                # Check convergence
                if np.all(np.abs(step_sizes) < 1e-8):
                    break

                x_old = x_curr.copy()
                f_old = f_curr
                delta_f = 0.0
                min_delta = 0.0
                min_dir = 0

                # Minimize along each direction
                for i in range(dim):
                    if total_calls - start_calls >= max_calls or total_calls >= budget:
                        break

                    # Line search along direction i
                    x_line = x_curr + step_sizes[i] * directions[i]
                    x_line = np.clip(x_line, -5.0, 5.0)
                    f_line = safe_call(x_line)
                    if f_line is None:
                        break

                    if f_line < f_curr:
                        # Move in this direction
                        x_curr = x_line
                        f_curr = f_line
                        delta_f += f_old - f_line

                        # Try to extend the step
                        x_extend = x_curr + step_sizes[i] * directions[i]
                        x_extend = np.clip(x_extend, -5.0, 5.0)
                        f_extend = safe_call(x_extend)
                        if f_extend is not None and f_extend < f_curr:
                            x_curr = x_extend
                            f_curr = f_extend
                            delta_f += f_old - f_extend
                            step_sizes[i] *= 2.0
                        else:
                            # Contract the step
                            step_sizes[i] *= 0.5
                            # Try to find a better step by halving
                            x_half = x_curr - 0.5 * step_sizes[i] * directions[i]
                            x_half = np.clip(x_half, -5.0, 5.0)
                            f_half = safe_call(x_half)
                            if f_half is not None and f_half < f_curr:
                                x_curr = x_half
                                f_curr = f_half
                                delta_f += f_old - f_half
                    else:
                        # No improvement, contract step
                        step_sizes[i] *= 0.5

                    if f_curr < best_f_powell:
                        best_f_powell = f_curr
                        best_x_powell = x_curr.copy()

                    # Track the direction with the largest improvement
                    if f_old - f_curr > min_delta:
                        min_delta = f_old - f_curr
                        min_dir = i

                # Check if we made any improvement
                if f_curr < best_f - 1e-8:
                    no_improve = 0
                    prev_best = best_f
                else:
                    no_improve += 1
                    if no_improve > 20:
                        break

                # Update directions: replace the worst direction with the new one
                # The new direction is the overall displacement
                new_dir = x_curr - x_old
                if np.linalg.norm(new_dir) > 1e-10:
                    # Normalize the new direction
                    new_dir = new_dir / np.linalg.norm(new_dir)
                    directions[min_dir] = new_dir

                # Check convergence based on step sizes
                if np.max(step_sizes) < 1e-6:
                    break

            return best_x, best_f

        # --- Multi-Start Strategy ---
        # We'll do multiple restarts with different initial points and step sizes
        rng = np.random.default_rng(42)

        # Allocate budget: reserve some for final refinement
        reserve = 50
        main_budget = budget - reserve

        # Define starting points and initial steps
        starts = [
            (np.zeros(dim), 2.0),
            (rng.uniform(-4.0, 4.0, dim), 2.0),
            (rng.uniform(-3.0, 3.0, dim), 1.5),
            (rng.uniform(-5.0, 5.0, dim), 2.5),
            (rng.uniform(-2.0, 2.0, dim), 1.0),
        ]

        calls_per_start = max(10, main_budget // len(starts))

        for x0, step in starts:
            if total_calls >= budget or time.time() > deadline - 10:
                break
            f0 = safe_call(x0)
            if f0 is None:
                break
            powell_method(x0, f0, calls_per_start, initial_step=step)

        # Final refinement on the best point found
        remaining = budget - total_calls
        if remaining > 10 and total_calls < budget:
            powell_method(best_x.copy(), best_f, remaining, initial_step=0.3)

        return best_x, best_f
