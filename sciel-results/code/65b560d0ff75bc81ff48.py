"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Powell's conjugate direction method for the Bent Cigar function.
    Maintains a set of linearly independent search vectors, updating them
    to align with the function's principal axes after each cycle.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        dim = self.dim
        calls_used = 0
        best_fx = float('inf')
        best_x = np.zeros(dim)

        def track(x):
            nonlocal calls_used, best_fx, best_x
            fx = func(x)
            calls_used += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Initialize at origin
        x = np.zeros(dim)
        fx = track(x)

        # Initialize search directions as standard basis
        directions = [np.eye(dim)[i] for i in range(dim)]

        # Powell's method
        while calls_used < self.budget and time.time() < deadline:
            x_old = x.copy()
            f_old = fx
            x_new = x_old.copy()
            f_new = fx
            max_delta = 0.0
            max_dir_idx = 0

            # Move along each direction
            for i in range(dim):
                if calls_used >= self.budget or time.time() > deadline:
                    break

                # Find the minimum along direction[i] from x_new
                # Use a simple line search: try several step sizes
                d = directions[i]
                # Initial step estimate
                step = 1.0
                # Try to find a good step by trying a few candidates
                best_step = 0.0
                best_f = f_new
                best_x_try = x_new.copy()

                # Try steps: negative and positive
                for sign in [-1, 1]:
                    for mult in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
                        s = sign * mult * step
                        x_try = np.clip(x_new + s * d, -5, 5)
                        f_try = track(x_try)
                        if f_try < best_f:
                            best_f = f_try
                            best_step = s
                            best_x_try = x_try.copy()

                if best_step != 0.0:
                    delta = best_f - f_new
                    f_new = best_f
                    x_new = best_x_try
                    if abs(delta) > abs(max_delta):
                        max_delta = delta
                        max_dir_idx = i

            # Check if we made progress
            if abs(max_delta) < 1e-12:
                # No progress, try a different strategy: restart with random perturbation
                break

            # Replace the direction that gave the largest improvement
            # with the direction from x_old to x_new
            new_direction = x_new - x_old
            if np.linalg.norm(new_direction) > 1e-15:
                directions[max_dir_idx] = new_direction / np.linalg.norm(new_direction)

            # Move to the new point
            x = x_new
            fx = f_new

            # If no significant progress, try a global restart
            if abs(max_delta) < 1e-10 and calls_used < self.budget * 0.8:
                # Reset directions to standard basis and try a random jump
                rng = np.random.default_rng(42 + calls_used)
                jump = rng.standard_normal(dim) * 2.0
                x_jump = np.clip(x + jump, -5, 5)
                f_jump = track(x_jump)
                if f_jump < fx:
                    x = x_jump
                    fx = f_jump
                directions = [np.eye(dim)[i] for i in range(dim)]

        return best_x, best_fx
# EVOLVE-BLOCK-END
