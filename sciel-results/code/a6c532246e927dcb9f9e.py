"""Rewritten optimizer for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit linear structure of f5: estimate gradient, jump to boundary."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Phase 1: Estimate gradient using coordinate differences
        # f5 is linear: f(x) = c + g^T (x - x0) after rotation/shift
        # We estimate the gradient direction by finite differences from origin
        origin = np.zeros(dim)
        f0 = func(origin)

        # Evaluate along each axis to get partial derivatives
        step = 1.0
        grad = np.zeros(dim)
        calls = 1  # f0

        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            e_i = np.zeros(dim)
            e_i[i] = step
            f_plus = func(e_i)
            calls += 1
            grad[i] = (f_plus - f0) / step

        # Also check negative direction for better estimate
        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            e_i = np.zeros(dim)
            e_i[i] = -step
            f_minus = func(e_i)
            calls += 1
            grad[i] = 0.5 * (f_plus - f_minus) / (2 * step)

        # Phase 2: Jump to the best corner in the direction of -grad
        # For linear function, minimum is at a boundary vertex
        # The direction to minimize is -grad
        norm_g = np.linalg.norm(grad)
        if norm_g < 1e-12:
            # Function is nearly constant; return origin
            return origin, f0

        # Try all 2^dim corners? No, too many. Instead, go far in -grad direction
        # and clip to box. For a linear function, the minimum over the box is at
        # a vertex. The best vertex is the one where x_i = -5 if g_i > 0, else +5.
        # Actually: minimize c + g^T x => for each i, if g_i > 0 set x_i = -5, else x_i = 5
        best_x = np.where(grad > 0, -5.0, 5.0)
        f_best = func(best_x)
        calls += 1

        # Phase 3: Refine with local search if budget remains
        if calls < budget:
            # Do a few rounds of coordinate descent / hill climbing
            x = best_x.copy()
            fx = f_best
            sigma = 1.0
            for _ in range(budget - calls):
                if time.time() > deadline:
                    break
                # Random perturbation
                y = x + sigma * np.random.default_rng().standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.8
                if sigma < 1e-6:
                    break

        return x, fx
# EVOLVE-BLOCK-END
