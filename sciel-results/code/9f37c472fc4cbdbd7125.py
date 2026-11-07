"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio-based continuous optimizer with time guard for ill-conditioned functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        def timed_call(x):
            return func(x)

        # Strategy 1: (1+lambda)-ES with adaptive sigma
        def run_1lambdees(budget_remaining, rng, x0, fx0):
            x = x0.copy()
            fx = fx0
            sigma = 1.5
            lam = 4
            success_count = 0
            total_trials = 0
            target_success_rate = 1.0 / (lam + 1)

            for i in range(budget_remaining):
                if not time_ok():
                    break
                # Generate offspring
                best_y = None
                best_fy = fx
                for j in range(lam):
                    if not time_ok():
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = timed_call(y)
                    if fy < best_fy:
                        best_fy = fy
                        best_y = y.copy()

                total_trials += lam
                if best_y is not None and best_fy < fx:
                    x = best_y
                    fx = best_fy
                    success_count += 1
                    sigma *= 1.15
                else:
                    sigma *= 0.85

                # Adaptive sigma adjustment based on success rate
                if total_trials > 20:
                    current_rate = success_count / total_trials
                    if current_rate > target_success_rate * 1.5:
                        sigma *= 1.05
                    elif current_rate < target_success_rate * 0.5:
                        sigma *= 0.95

            return x, fx

        # Strategy 2: CMA-ES-like with diagonal covariance
        def run_cma_simple(budget_remaining, rng, x0, fx0):
            x = x0.copy()
            fx = fx0
            sigma = 1.0
            C = np.eye(dim)
            # Track step sizes per dimension
            step_sizes = np.ones(dim)
            success_history = []

            for i in range(budget_remaining):
                if not time_ok():
                    break
                lam = 4
                best_y = None
                best_fy = fx

                for j in range(lam):
                    if not time_ok():
                        break
                    # Sample from diagonal normal
                    z = rng.standard_normal(dim)
                    y = np.clip(x + sigma * step_sizes * z, -5, 5)
                    fy = timed_call(y)
                    if fy < best_fy:
                        best_fy = fy
                        best_y = y.copy()

                if best_y is not None and best_fy < fx:
                    # Update step sizes based on direction
                    direction = best_y - x
                    for d in range(dim):
                        if abs(direction[d]) > 1e-10:
                            step_sizes[d] *= 1.1
                    x = best_y
                    fx = best_fy
                    success_history.append(1)
                else:
                    for d in range(dim):
                        step_sizes[d] *= 0.9
                    success_history.append(0)

                # Normalize step sizes to prevent explosion
                mean_step = np.mean(step_sizes)
                if mean_step > 5.0:
                    step_sizes *= 0.5
                elif mean_step < 0.01:
                    step_sizes *= 2.0

                # Adaptive sigma
                if len(success_history) > 10:
                    recent = success_history[-10:]
                    rate = np.mean(recent)
                    if rate > 0.4:
                        sigma *= 1.05
                    elif rate < 0.1:
                        sigma *= 0.95

            return x, fx

        # Main loop: try different strategies with restarts
        rng = np.random.default_rng(0)
        best_x = None
        best_f = np.inf

        # Initial evaluation
        x_init = rng.uniform(-5, 5, dim)
        f_init = timed_call(x_init)
        best_x = x_init.copy()
        best_f = f_init

        # Phase 1: (1+lambda)-ES
        budget_phase1 = min(200, budget - 1)
        x1, f1 = run_1lambdees(budget_phase1, rng, best_x.copy(), best_f)
        if f1 < best_f:
            best_x, best_f = x1, f1

        # Phase 2: CMA-like refinement
        if time_ok():
            budget_phase2 = min(150, budget - 1 - budget_phase1)
            x2, f2 = run_cma_simple(budget_phase2, rng, best_x.copy(), best_f)
            if f2 < best_f:
                best_x, best_f = x2, f2

        # Phase 3: Final polish with small steps
        if time_ok():
            remaining = budget - 1 - budget_phase1 - budget_phase2
            if remaining > 0:
                sigma = 0.1
                for i in range(min(remaining, 50)):
                    if not time_ok():
                        break
                    y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = timed_call(y)
                    if fy < best_f:
                        best_x = y.copy()
                        best_f = fy
                        sigma *= 1.2
                    else:
                        sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
