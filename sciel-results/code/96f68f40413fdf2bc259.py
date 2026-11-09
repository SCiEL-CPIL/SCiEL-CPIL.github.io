"""Model-and-optimize approach for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB-style f5 (Linear slope).
    
    Strategy:
    1. Estimate the linear model f(x) ≈ c + g·x using centered differences at origin.
    2. Predict the minimizer of the linear model (box vertex) and jump there.
    3. Refine with (1+1)-ES using the remaining budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        budget_used = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def record(x, f):
            nonlocal best_f, best_x
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Estimate linear model at origin using centered differences
        # f(0) gives the offset c
        x0 = np.zeros(dim)
        f0 = func(x0)
        budget_used += 1
        record(x0, f0)

        # Centered difference step size
        h = 0.5
        grad = np.zeros(dim)

        for i in range(dim):
            if time.time() > deadline:
                break
            e_i = np.zeros(dim)
            e_i[i] = 1.0

            x_plus = np.clip(x0 + h * e_i, -5, 5)
            x_minus = np.clip(x0 - h * e_i, -5, 5)
            f_plus = func(x_plus)
            budget_used += 1
            record(x_plus, f_plus)
            f_minus = func(x_minus)
            budget_used += 1
            record(x_minus, f_minus)

            grad[i] = (f_plus - f_minus) / (2.0 * h)

        # Phase 2: Jump to predicted minimizer of the linear model
        # For f(x) ≈ c + g·x on [-5,5]^d, min is at x_i = -5*sign(g_i)
        x_pred = np.zeros(dim)
        for i in range(dim):
            if grad[i] > 1e-8:
                x_pred[i] = -5.0
            elif grad[i] < -1e-8:
                x_pred[i] = 5.0
            else:
                x_pred[i] = 0.0

        if budget_used < budget:
            f_pred = func(x_pred)
            budget_used += 1
            record(x_pred, f_pred)

        # Phase 3: Local refinement with (1+1)-ES
        x = best_x.copy()
        fx = best_f
        # Start with a small step since we're likely near the optimum
        sigma = 0.3

        fail_count = 0
        max_fails = 30

        while budget_used < budget:
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            budget_used += 1

            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.05, 2.0)
                fail_count = 0
                record(x, fx)
            else:
                sigma = max(sigma * 0.95, 0.01)
                fail_count += 1

            if fail_count > max_fails:
                sigma = max(sigma, 0.05)
                fail_count = 0

            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
