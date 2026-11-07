"""Bootstrap spawn: MODEL-AND-OPTIMIZE approach for linear slope f5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct linear model fit for f5 (Linear slope), with minimal polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        best_x = None
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, call_count
            if time.time() > deadline:
                return best_f
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        dim = self.dim

        # Phase 1: Robust sign determination using ±1 probes (interior points)
        # Evaluate at center
        x_center = np.zeros(dim)
        f_center = evaluate(x_center)

        # For each dimension, probe at +1 and -1 to robustly determine gradient sign
        # Using interior points avoids boundary clipping effects
        step = 1.0
        signs = np.zeros(dim)

        for i in range(dim):
            if time.time() > deadline:
                break
            x_plus = x_center.copy()
            x_plus[i] += step
            f_plus = evaluate(x_plus)

            x_minus = x_center.copy()
            x_minus[i] -= step
            f_minus = evaluate(x_minus)

            # Gradient sign: if f_plus > f_center, gradient is positive (function increases in + direction)
            # So minimum is in - direction. If f_minus > f_center, gradient is negative, minimum in + direction.
            # Use both probes for robustness: compare f_plus and f_minus
            if f_plus > f_minus:
                # Function increases more in + direction, so gradient is positive
                signs[i] = 1.0
            else:
                # Function increases more in - direction (or decreases in +), gradient is negative
                signs[i] = -1.0

        # Compute the linear model optimum (vertex of the box)
        # For each dim: if gradient > 0 (sign=1), choose -5; else choose +5
        x_vertex = np.where(signs > 0, -5.0, 5.0)
        f_vertex = evaluate(x_vertex)

        # Phase 2: Verify with a few probes near the vertex
        # If the function is truly linear, the vertex is the global minimum.
        # Do a small number of verification calls to confirm.
        if call_count < self.budget and time.time() < deadline:
            x = x_vertex.copy()
            fx = f_vertex
            sigma = 0.1

            rng = np.random.default_rng(123)

            for _ in range(min(10, self.budget - call_count)):
                if time.time() > deadline:
                    break

                direction = rng.standard_normal(dim)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm
                y = x + sigma * direction
                y = np.clip(y, -5.0, 5.0)

                fy = evaluate(y)

                if fy < fx - 1e-10:
                    x = y
                    fx = fy
                    sigma *= 1.2
                else:
                    sigma *= 0.5

                if sigma < 1e-5:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
