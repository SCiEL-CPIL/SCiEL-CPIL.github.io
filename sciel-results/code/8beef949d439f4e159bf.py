"""Multi-restart (1+1)-ES with coordinate-descent polish for Schwefel f20."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-restart (1+1)-ES with coordinate-descent polish on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # safety margin before 330s kill
        calls = 0

        def F(x):
            nonlocal calls
            calls += 1
            return func(x)

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf

        def time_ok():
            return time.time() < deadline - 3.0

        def run_es(x0, f0, budget_alloc):
            """Run (1+1)-ES from x0 with given budget."""
            nonlocal best_x, best_f, calls
            n = dim
            x = x0.copy()
            fx = f0
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            sigma = 1.0
            success_count = 0
            total_trials = 0
            no_improve = 0

            while calls < budget_alloc and time_ok():
                z = rng.standard_normal(n)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = F(y)
                total_trials += 1

                if fy < fx:
                    x = y.copy()
                    fx = fy
                    success_count += 1
                    no_improve = 0
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    no_improve += 1

                # 1/5 success rule
                if total_trials >= 5:
                    if success_count / total_trials > 1.0 / 5:
                        sigma *= 1.25
                    else:
                        sigma *= 0.8
                    success_count = 0
                    total_trials = 0

                sigma = np.clip(sigma, 1e-12, 10.0)

                # Escape: if stuck for many iterations, jump
                if no_improve > 30:
                    x = x + rng.standard_normal(n) * 1.0
                    x = np.clip(x, -5, 5)
                    fx = F(x)
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    no_improve = 0
                    sigma = max(sigma, 0.5)

            return x, fx

        def coord_descent_polish(x0, f0, budget_alloc):
            """Coordinate descent with adaptive steps."""
            nonlocal best_x, best_f, calls
            n = dim
            x = x0.copy()
            fx = f0
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            steps = np.full(n, 0.1)
            min_step = 1e-8

            while calls < budget_alloc and time_ok():
                improved = False
                for d in range(n):
                    if calls >= budget_alloc or not time_ok():
                        break
                    for sign in [1, -1]:
                        if calls >= budget_alloc or not time_ok():
                            break
                        y = x.copy()
                        y[d] += sign * steps[d]
                        y = np.clip(y, -5, 5)
                        fy = F(y)
                        if fy < fx:
                            x = y.copy()
                            fx = fy
                            improved = True
                            if fx < best_f:
                                best_f, best_x = fx, x.copy()
                        else:
                            steps[d] *= 0.7

                if not improved:
                    steps *= 0.5
                    if np.all(steps < min_step):
                        break

            return x, fx

        # Phase 1: Multi-restart (1+1)-ES
        # Use ~70% of budget for multiple ES restarts
        es_budget_total = int(budget * 0.7)
        n_restarts = 8
        budget_per_restart = es_budget_total // n_restarts

        # Generate diverse starting points
        start_points = []
        # Center
        start_points.append(np.zeros(dim))
        # Random points in [-4, 4]^dim
        for _ in range(n_restarts - 1):
            start_points.append(rng.uniform(-4, 4, dim))

        for i, sp in enumerate(start_points):
            if calls >= es_budget_total or not time_ok():
                break
            remaining_for_this = min(budget_per_restart, es_budget_total - calls)
            if remaining_for_this < 5:
                break
            # Evaluate starting point
            f_sp = F(sp)
            if f_sp < best_f:
                best_f, best_x = f_sp, sp.copy()
            # Run ES from this point
            run_es(sp, f_sp, remaining_for_this)

        # Phase 2: Coordinate descent polish from best
        remaining = budget - calls
        if remaining > 10 and time_ok():
            coord_descent_polish(best_x.copy(), best_f, remaining)

        return best_x, best_f
# EVOLVE-BLOCK-END
