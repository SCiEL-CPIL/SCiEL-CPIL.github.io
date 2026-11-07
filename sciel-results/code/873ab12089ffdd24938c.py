"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start adaptive (1+1)-ES with LHS initialization and local restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        rng = np.random.default_rng(12345)

        calls_made = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # LHS initialization: spread initial points across the box
        n_lhs = min(15, budget // 10)
        lhs_points = []
        for _ in range(n_lhs):
            strata = np.random.permutation(n_lhs)
            point = (strata + rng.uniform(0, 1, dim)) / n_lhs * 10.0 - 5.0
            lhs_points.append(point)

        # Evaluate initial points
        for p in lhs_points:
            evaluate(p)

        # Also evaluate center
        evaluate(np.zeros(dim))

        remaining = budget - calls_made
        if remaining <= 0:
            return best_x, best_f

        # Adaptive (1+1)-ES with 1/5 success rule
        def run_es(start_x, start_f, num_calls):
            nonlocal calls_made, best_x, best_f
            x = start_x.copy()
            fx = start_f
            sigma = 1.5
            for i in range(num_calls):
                if time.time() > deadline or calls_made >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.9
                    if sigma < 0.01:
                        sigma = 0.01
            return x, fx, sigma

        # Run ES from best point with most of the budget
        main_budget = int(remaining * 0.8)
        x, fx, sigma = run_es(best_x.copy(), best_f, main_budget)

        if time.time() > deadline or calls_made >= budget:
            return best_x, best_f

        # Try a few restarts from different regions
        for _ in range(5):
            if time.time() > deadline or calls_made >= budget:
                break
            # Generate a point in a different region
            restart_x = rng.uniform(-5, 5, dim)
            if np.linalg.norm(restart_x - best_x) < 2.0:
                restart_x = -best_x + rng.uniform(-1, 1, dim)
                restart_x = np.clip(restart_x, -5, 5)
            rf = evaluate(restart_x)
            # Run a short ES from this point
            if budget - calls_made > 10:
                run_es(restart_x.copy(), rf, min(20, budget - calls_made))

        return best_x, best_f
# EVOLVE-BLOCK-END
