"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start adaptive (1+1)-ES with stagnation-driven restarts for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270  # 30s safety margin before 300s limit

        # Stratified initialization: mix of global and local scales
        # First point: center
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Multi-start exploration phase: evaluate several diverse points
        explore_count = min(20, budget // 5)
        for _ in range(explore_count - 1):
            if time.time() > deadline:
                break
            # Mix of scales: some global, some medium, some local
            scale = rng.choice([4.0, 2.0, 1.0, 0.5])
            x = rng.uniform(-5, 5, n) * scale
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Exploitation phase: adaptive (1+1)-ES from best point
        x = best_x.copy()
        fx = best_fx
        sigma = 1.0

        # Stagnation tracking
        no_improve = 0
        stagnation_threshold = 30
        restart_count = 0
        max_restarts = 10

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                # Success: move and increase sigma
                x, fx = y, fy
                sigma *= 1.20
                no_improve = 0
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                # Failure: decrease sigma
                sigma *= 0.75
                no_improve += 1

                # Stagnation: restart from a perturbation of best point
                if no_improve >= stagnation_threshold and restart_count < max_restarts:
                    restart_count += 1
                    # Jump scale decreases with restart count
                    jump_scale = max(0.5, 3.0 * (0.5 ** restart_count))
                    x = np.clip(best_x + jump_scale * rng.standard_normal(n), -5, 5)
                    fx = func(x)
                    calls += 1
                    sigma = jump_scale * 0.5
                    no_improve = 0
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()

            # Fine-tuning: if sigma is very small, use smaller steps
            if sigma < 0.01:
                sigma = 0.01

        return best_x, best_fx
# EVOLVE-BLOCK-END
