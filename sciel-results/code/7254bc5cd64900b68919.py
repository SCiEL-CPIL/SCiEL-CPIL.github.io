"""Rewritten program: exploit linear structure via gradient estimation and corner jump."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear-function optimizer: estimate gradient, jump to optimal corner."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        d = self.dim
        budget = self.budget

        # Phase 1: Estimate gradient using finite differences at center
        # For a linear function f(x) = c + g^T x, we can estimate g precisely
        # by evaluating at center and along each axis.
        center = np.zeros(d)
        f_center = func(center)
        budget_used = 1

        # Probe each axis with a large step to capture full gradient magnitude
        h = 5.0  # full box half-width
        grad = np.zeros(d)
        for i in range(d):
            if budget_used >= budget:
                break
            xi = center.copy()
            xi[i] = h
            fi = func(xi)
            budget_used += 1
            grad[i] = (fi - f_center) / h

        # Phase 2: The minimum of a linear function on [-5,5]^d is at a vertex.
        # The optimal vertex has x_i = -5 if grad[i] > 0, else x_i = 5 if grad[i] < 0.
        # If grad[i] == 0, either works.
        corner = np.where(grad > 0, -5.0, 5.0)
        f_corner = func(corner)
        budget_used += 1

        # Track best
        x_best = corner.copy()
        f_best = f_corner

        # Phase 3: Use remaining budget to refine with local search
        # For a truly linear function, the corner IS the optimum, so any further
        # evaluations won't improve. But to be safe and handle any nonlinearity,
        # do a few local steps.
        sigma = 0.5
        for _ in range(budget - budget_used):
            if time.time() > deadline:
                break
            y = np.clip(x_best + sigma * np.random.default_rng(0).standard_normal(d), -5, 5)
            fy = func(y)
            if fy < f_best:
                x_best = y
                f_best = fy
                sigma *= 1.2
            else:
                sigma *= 0.8
            if sigma < 1e-6:
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
