"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Powell's conjugate direction method with restarts for the Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        n_calls = 0
        dim = self.dim
        budget = self.budget

        # We will track the best point found overall
        best_x = None
        best_f = np.inf

        def call_func(x):
            nonlocal n_calls, best_x, best_f
            n_calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Powell's method: minimize along a set of directions
        # The Bent Cigar is extremely anisotropic, so Powell's method should
        # quickly align with the principal axes.

        def powell_cycle(x, fx, directions, n_calls_ref):
            """Perform one cycle of Powell's method. Returns new x, fx, updated directions."""
            dim = len(x)
            n = dim
            x_prev = x.copy()
            fx_prev = fx
            max_diff = -np.inf
            max_i = 0

            for i in range(n):
                # Minimize along direction[i]
                # We use a simple 1D search: try steps in both directions
                d = directions[i]
                # Normalize direction for step size
                d_norm = np.linalg.norm(d)
                if d_norm < 1e-15:
                    continue
                d = d / d_norm

                # Try a few step sizes to find a good improvement
                # Use a simple approach: try decreasing step sizes
                best_step = 0.0
                best_val = fx
                # Start with a reasonable step
                step = 1.0
                for _ in range(20):
                    # Try positive and negative steps
                    for sign in [1, -1]:
                        x_trial = np.clip(x + sign * step * d, -5, 5)
                        ft = call_func(x_trial)
                        if ft < best_val:
                            best_val = ft
                            best_step = sign * step
                    step *= 0.5
                    if step < 1e-6:
                        break

                if best_step != 0.0:
                    x = np.clip(x + best_step * d, -5, 5)
                    fx = best_val

                diff = fx_prev - fx
                if diff > max_diff:
                    max_diff = diff
                    max_i = i

            # Check if cycle made progress
            if max_diff <= 1e-10:
                return x, fx, directions

            # Powell update: replace the direction that gave least improvement
            # with the direction from x_prev to x
            new_dir = x - x_prev
            # Replace direction[max_i] with new_dir
            # But we need to maintain linear independence; for now, just replace
            # The standard Powell update:
            # - The direction that gave the least improvement is replaced by (x - x_prev)
            # - Actually, standard Powell replaces the direction with the SMALLEST improvement
            # Let me re-read: in Powell's method, you track which direction gave the
            # LARGEST improvement (max_i), and you replace the direction with the
            # SMALLEST improvement. Let me fix this.

            # Actually, I need to track the smallest improvement, not largest.
            # Let me redo this properly.

            return x, fx, directions

        # Let me implement a cleaner version of Powell's method

        def powell_minimize(x0, fx0, max_iters, n_calls_ref):
            """Powell's conjugate direction method."""
            dim = len(x0)
            n = dim

            # Initialize directions as standard basis
            directions = np.eye(n)
            x = x0.copy()
            fx = fx0

            for iteration in range(max_iters):
                if time.time() > deadline:
                    break

                x_prev = x.copy()
                fx_prev = fx
                min_diff = np.inf
                min_i = 0

                for i in range(n):
                    if time.time() > deadline:
                        break

                    d = directions[i]
                    d_norm = np.linalg.norm(d)
                    if d_norm < 1e-15:
                        continue
                    d = d / d_norm

                    # 1D line search along direction d
                    best_step = 0.0
                    best_val = fx

                    # Try a range of step sizes
                    step = 2.0
                    for _ in range(30):
                        for sign in [1, -1]:
                            x_trial = np.clip(x + sign * step * d, -5, 5)
                            ft = call_func(x_trial)
                            if ft < best_val:
                                best_val = ft
                                best_step = sign * step
                        step *= 0.5
                        if step < 1e-7:
                            break

                    if best_step != 0.0:
                        x = np.clip(x + best_step * d, -5, 5)
                        fx = best_val

                    diff = fx_prev - fx
                    if diff < min_diff:
                        min_diff = diff
                        min_i = i

                # If no improvement in this cycle, break
                if fx >= fx_prev - 1e-12:
                    break

                # Powell update: replace direction[min_i] with (x - x_prev)
                new_dir = x - x_prev
                directions[min_i] = new_dir

                # Optional: do an extra step along the new direction
                d_new = new_dir
                d_norm = np.linalg.norm(d_new)
                if d_norm > 1e-15:
                    d_new = d_new / d_norm
                    # Line search along new direction
                    best_step = 0.0
                    best_val = fx
                    step = 1.0
                    for _ in range(20):
                        for sign in [1, -1]:
                            x_trial = np.clip(x + sign * step * d_new, -5, 5)
                            ft = call_func(x_trial)
                            if ft < best_val:
                                best_val = ft
                                best_step = sign * step
                        step *= 0.5
                        if step < 1e-7:
                            break
                    if best_step != 0.0:
                        x = np.clip(x + best_step * d_new, -5, 5)
                        fx = best_val

            return x, fx

        # Multi-start strategy with Powell's method
        # The Bent Cigar has a narrow valley, so we need good initialization

        # First, do a coarse random search to find a decent starting point
        # Then apply Powell's method

        # Generate several starting points
        rng = np.random.default_rng(42)

        # Try origin and a few random points
        candidates = [np.zeros(dim)]
        for _ in range(10):
            candidates.append(rng.uniform(-4, 4, dim))

        # Evaluate each candidate
        best_start = None
        best_start_f = np.inf
        for c in candidates:
            fc = call_func(c)
            if fc < best_start_f:
                best_start_f = fc
                best_start = c

        # Apply Powell's method from the best starting point
        if time.time() < deadline and n_calls < budget:
            x_p, fx_p = powell_minimize(best_start, best_start_f, max_iters=100, n_calls_ref=0)

        # If we still have budget, try a few more restarts from different points
        restarts = 0
        while time.time() < deadline and n_calls < budget - 10 and restarts < 5:
            # Generate a new random starting point
            x_new = rng.uniform(-4, 4, dim)
            fx_new = call_func(x_new)

            # Only do Powell if this is competitive
            if fx_new < best_f + 10:
                x_p, fx_p = powell_minimize(x_new, fx_new, max_iters=50, n_calls_ref=0)
            restarts += 1

        return best_x, best_f

# EVOLVE-BLOCK-END
