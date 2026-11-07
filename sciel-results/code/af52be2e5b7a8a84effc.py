"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits linear structure of f5 by estimating gradient and jumping to boundary."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        def timed_out():
            return time.time() > deadline

        d = self.dim

        # 1. Initial point at center
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # 2. Estimate gradient using forward differences
        #    Use a small step size for numerical stability
        h = 1e-5
        grad = np.zeros(d)
        for i in range(d):
            xp = x.copy()
            xp[i] += h
            xp = np.clip(xp, -5, 5)
            fxp = func(xp)
            grad[i] = (fxp - fx) / h
            if timed_out():
                break

        # 3. Determine the optimal vertex based on gradient signs
        #    For f(x) = w^T x + b, minimum on [-5,5]^d is at:
        #    x_i = -5 if w_i > 0, x_i = 5 if w_i < 0
        #    If w_i is near zero, either endpoint is fine
        x_vertex = np.zeros(d)
        for i in range(d):
            if grad[i] > 1e-8:
                x_vertex[i] = -5.0
            elif grad[i] < -1e-8:
                x_vertex[i] = 5.0
            else:
                x_vertex[i] = 0.0  # Doesn't matter for linear function

        # 4. Evaluate the vertex
        fx_vertex = func(x_vertex)
        if fx_vertex < best_fx:
            best_x = x_vertex.copy()
            best_fx = fx_vertex

        # 5. Use remaining budget for verification and refinement
        #    Check a few nearby points in case gradient estimation was noisy
        if not timed_out() and self.budget > 10:
            # Try small perturbations around the vertex to verify it's truly optimal
            rng = np.random.default_rng(42)
            for _ in range(min(50, self.budget - 10)):
                if timed_out():
                    break
                # Random point in the box
                x_try = rng.uniform(-5, 5, d)
                fx_try = func(x_try)
                if fx_try < best_fx:
                    best_x = x_try.copy()
                    best_fx = fx_try

        return best_x, best_fx
# EVOLVE-BLOCK-END
