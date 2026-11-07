"""Rewritten program: gradient-based optimizer for linear-slope BBOB f5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Gradient-based optimizer for the linear-slope function f5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        deadline = time.time() + 290

        rng = np.random.default_rng(0)

        # Start at origin
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Estimate gradient using central differences
        h = 1e-4
        grad = np.zeros(d)
        for i in range(d):
            xp = x.copy()
            xp[i] += h
            xm = x.copy()
            xm[i] -= h
            grad[i] = (func(xp) - func(xm)) / (2 * h)
            calls += 2

        # For linear f(x) = a^T x + b, the minimum on [-5,5]^d is at the vertex
        # x_i = -5 if a_i > 0, x_i = 5 if a_i < 0
        a = grad
        x_vertex = np.where(a > 0, -5.0, 5.0)
        x_vertex[np.abs(a) < 1e-12] = 0.0

        fx_vertex = func(x_vertex)
        calls += 1

        if fx_vertex < best_fx:
            best_x = x_vertex.copy()
            best_fx = fx_vertex

        # Use remaining budget for a fine local search around the vertex
        sigma = 0.05
        for _ in range(budget - calls):
            if time.time() > deadline:
                break
            y = np.clip(x_vertex + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls += 1
            if fy < best_fx:
                best_x = y.copy()
                best_fx = fy
            sigma *= 0.97
            if sigma < 1e-10:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
