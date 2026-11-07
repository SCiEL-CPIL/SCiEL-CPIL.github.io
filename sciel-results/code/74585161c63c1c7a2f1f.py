"""Optimizer for BBOB-style f12 (Bent cigar) with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with restarts for bent cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin (limit is 300s)
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Best result so far
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_used = 0

        # Multi-restart strategy with different initializations
        # The bent cigar function has a narrow valley, so we need good direction
        # Try several restarts with different strategies

        def evaluate(x):
            nonlocal calls_used
            calls_used += 1
            return func(x)

        # Strategy 1: Start near origin (minimizer is in [-4,4]^5)
        # Use a sequence of restarts with different step sizes and directions

        max_restarts = 8
        restarts_done = 0

        for restart in range(max_restarts):
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break

            # Different initialization strategies per restart
            if restart == 0:
                x = np.zeros(dim)
            elif restart == 1:
                x = rng.uniform(-2, 2, dim)
            elif restart == 2:
                x = rng.uniform(-4, 4, dim)
            elif restart == 3:
                # Try along a random direction
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction)
                x = 3.0 * direction
            elif restart == 4:
                x = rng.uniform(-5, 5, dim)
            elif restart == 5:
                # Small perturbation around best
                x = best_x + 0.5 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
            elif restart == 6:
                x = rng.uniform(-1, 1, dim)
            else:
                x = rng.uniform(-3, 3, dim)

            fx = evaluate(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive step size
            sigma = 1.0
            # For bent cigar, we need to adapt quickly
            # Use a more aggressive adaptation

            # Track success rate for step size adaptation
            successes = 0
            total_steps = 0
            window = 10

            for _ in range(budget - calls_used):
                if time.time() > deadline:
                    break

                # Generate trial point
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = evaluate(y)

                total_steps += 1
                if fy < fx:
                    successes += 1
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Increase step size on success
                    sigma *= 1.3
                else:
                    # Decrease step size on failure
                    sigma *= 0.85

                # Prevent sigma from getting too large or too small
                sigma = np.clip(sigma, 1e-4, 50.0)

                # Check for convergence: if no improvement in many steps, restart
                if total_steps % window == 0:
                    success_rate = successes / window
                    if success_rate < 0.1 and total_steps > 50:
                        break
                    successes = 0

            restarts_done += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
