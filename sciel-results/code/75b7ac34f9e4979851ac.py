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
        deadline = start_time + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls_made = 0
        max_calls = self.budget  # 500

        def elapsed_ok():
            return time.time() < deadline and calls_made < max_calls

        # Initial exploration: sample a few random points to get a decent start
        # We'll do a small initial population to find a good starting point
        pop_size = 8
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

        # Adaptive step size parameters
        sigma = 2.0
        min_sigma = 0.005
        max_sigma = 6.0
        lambda_ = 8  # population size per generation
        stagnation_count = 0
        stagnation_limit = 25  # restart if no improvement for this many evaluations

        # Track best improvement for adaptive restarts
        best_improvement = np.inf
        last_improvement_call = 0
        no_improve_calls = 0
        last_best_f = best_f

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
                    improvement = fx - fy
                    x = y
                    fx = fy
                    improved = True
                    # Increase step size on success (more aggressive)
                    sigma = min(sigma * 1.3, max_sigma)
                    stagnation_count = 0
                    last_improvement_call = calls_made
                    no_improve_calls = 0

                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        if improvement < best_improvement:
                            best_improvement = improvement
                else:
                    # Decrease step size slightly on failure
                    sigma = max(sigma * 0.85, min_sigma)

            if not improved:
                stagnation_count += 1
                no_improve_calls += lambda_
            else:
                stagnation_count = 0

            # Periodic local refinement around best_x every ~40 calls without new best
            if no_improve_calls >= 40 and no_improve_calls < 60:
                if not elapsed_ok():
                    break
                # Try a few random perturbations around best_x with smaller steps
                for _ in range(3):
                    if not elapsed_ok():
                        break
                    perturbation = rng.standard_normal(self.dim) * 0.3
                    y = np.clip(best_x + perturbation, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        sigma = 1.0
                        stagnation_count = 0
                        last_improvement_call = calls_made
                        no_improve_calls = 0
                    elif fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        last_improvement_call = calls_made
                        no_improve_calls = 0

            # Reset stagnation counter if we made progress
            if improved:
                stagnation_count = 0

            # If we're very close to a likely minimum, do a bigger jump
            if sigma < 0.05 and not improved:
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
                    stagnation_count = 0
                    last_improvement_call = calls_made
                    no_improve_calls = 0
                elif fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    last_improvement_call = calls_made
                    no_improve_calls = 0

            # If we've been stuck for a long time, do a more aggressive restart
            if stagnation_count >= stagnation_limit:
                if not elapsed_ok():
                    break
                # Try multiple random restarts
                for _ in range(5):
                    if not elapsed_ok():
                        break
                    y = rng.uniform(-4.5, 4.5, self.dim)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        sigma = 2.0
                        stagnation_count = 0
                        last_improvement_call = calls_made
                        no_improve_calls = 0
                    elif fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        last_improvement_call = calls_made
                        no_improve_calls = 0

                # Reset stagnation counter
                stagnation_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
