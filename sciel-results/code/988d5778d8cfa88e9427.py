"""Rewritten optimizer for BBOB f7 (Step ellipsoidal).
Direct approach: robust population-based search with adaptive step size,
multiple restarts based on stagnation, and a strict time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based search with adaptive step size and stagnation restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # safety margin before 330s kill

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls_made = 0
        max_calls = self.budget  # 500

        def elapsed_ok():
            return time.time() < deadline and calls_made < max_calls

        # Initial exploration: sample fewer random points to save budget for search
        pop_size = 5
        candidates = []
        for _ in range(pop_size):
            if not elapsed_ok():
                break
            x = rng.uniform(-4.5, 4.5, self.dim)
            fx = func(x)
            calls_made += 1
            candidates.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort candidates and pick the best as starting point
        candidates.sort(key=lambda t: t[0])
        x = candidates[0][1].copy()
        fx = candidates[0][0]

        # Adaptive step size parameters - more aggressive adaptation
        sigma = 2.0
        min_sigma = 0.005
        max_sigma = 5.0
        lambda_ = 8  # smaller population for faster iteration
        stagnation_count = 0
        stagnation_limit = 25

        # Main search loop
        while elapsed_ok():
            # Generate offspring
            improved = False
            for j in range(lambda_):
                if not elapsed_ok():
                    break

                # Adaptive mutation
                step = rng.standard_normal(self.dim) * sigma
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    # Accept
                    x = y
                    fx = fy
                    improved = True
                    # Increase step size more aggressively on success
                    sigma = min(sigma * 1.3, max_sigma)
                    stagnation_count = 0

                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                else:
                    # Decrease step size on failure
                    sigma = max(sigma * 0.85, min_sigma)

            if not improved:
                stagnation_count += 1
            else:
                stagnation_count = 0

            # If stagnated, try local restart around best_x with varied scales
            if stagnation_count >= stagnation_limit // 2:
                if not elapsed_ok():
                    break
                # Try perturbations at multiple scales
                for scale in [0.3, 0.8, 1.5]:
                    if not elapsed_ok():
                        break
                    perturbation = rng.standard_normal(self.dim) * scale
                    y = np.clip(best_x + perturbation, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        sigma = max(0.5, scale)
                        stagnation_count = 0
                        break
                    elif fy < best_f:
                        best_f = fy
                        best_x = y.copy()

            # Reset stagnation counter if we made progress
            if improved:
                stagnation_count = 0

            # If sigma is very small and no improvement, do a bigger jump
            if sigma < 0.1 and not improved:
                # Big jump
                if not elapsed_ok():
                    break
                y = rng.uniform(-5, 5, self.dim)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x = y
                    fx = fy
                    sigma = 2.0
                elif fy < best_f:
                    best_f = fy
                    best_x = y.copy()

        # Final polish phase: if we have remaining budget, do fine-tuning near best
        remaining_calls = max_calls - calls_made
        if remaining_calls > 5 and elapsed_ok():
            # Use a fraction of remaining calls for polish
            polish_calls = min(remaining_calls, 30)
            polish_sigma = 0.05
            x = best_x.copy()
            fx = best_f
            for _ in range(polish_calls):
                if not elapsed_ok():
                    break
                step = rng.standard_normal(self.dim) * polish_sigma
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                else:
                    polish_sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
