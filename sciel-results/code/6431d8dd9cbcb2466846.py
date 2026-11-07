"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with coordinate descent polish for Schwefel-like functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 330s limit
        rng = np.random.default_rng(0)
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        sigma = 2.0

        # Track success rate for adaptive step size
        successes = 0
        window_size = 10
        recent_outcomes = []

        # Main loop: (1+1)-ES with periodic coordinate descent
        while calls < self.budget and time.time() < deadline:
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            fy = func(y)
            calls += 1

            if fy < fx:
                x = y
                fx = fy
                successes += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                successes -= 0  # no change, but we track in window

            recent_outcomes.append(1 if fy < fx else 0)
            if len(recent_outcomes) > window_size:
                recent_outcomes.pop(0)

            # Adaptive sigma based on success rate (1/5 rule approximation)
            if len(recent_outcomes) == window_size:
                success_rate = sum(recent_outcomes) / window_size
                if success_rate > 0.5:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.8
                sigma = max(0.01, min(sigma, 5.0))

            # Periodic coordinate descent polish every 50 calls
            if calls % 50 == 0 and calls < self.budget - 10:
                # Simple coordinate descent: try moving along each axis
                for i in range(self.dim):
                    if calls >= self.budget or time.time() >= deadline:
                        break
                    for step_dir in [-1, 1]:
                        if calls >= self.budget or time.time() >= deadline:
                            break
                        trial = x.copy()
                        trial[i] += step_dir * sigma * 0.1
                        trial[i] = np.clip(trial[i], -5.0, 5.0)
                        ft = func(trial)
                        calls += 1
                        if ft < fx:
                            x = trial
                            fx = ft
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
