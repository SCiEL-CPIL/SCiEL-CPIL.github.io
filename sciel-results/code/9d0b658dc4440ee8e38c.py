"""Optimizer for BBOB-style f15 (Rastrigin rotated) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start adaptive step-size local search with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin before the 330s kill
        D = self.dim
        B = self.budget

        # Count function calls to respect budget
        calls = [0]

        def f(x):
            if calls[0] >= B:
                return float('inf')
            calls[0] += 1
            return float(func(np.asarray(x, dtype=float)))

        rng = np.random.default_rng(42)

        # Generate diverse starting points
        n_starts = 8
        starts = [np.zeros(D)]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-5, 5, D))

        best_x = None
        best_f = np.inf

        # Phase 1: Multi-start with adaptive per-dimension step sizes
        # Use a (1+1)-ES-like approach from each start with adaptive steps
        base_sigma = 1.5
        success_threshold = 0.5  # 1/5 rule

        for s in starts:
            if time.time() >= deadline or calls[0] >= B:
                break

            x = np.array(s, dtype=float)
            fx = f(x)
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float)

            # Adaptive step sizes per dimension
            sigmas = np.full(D, base_sigma)
            successes = np.zeros(D)
            n_trials = 0
            max_local_iters = max(20, B // (2 * n_starts))

            for _ in range(max_local_iters):
                if time.time() >= deadline or calls[0] >= B:
                    break

                # Generate perturbation
                step = sigmas * rng.standard_normal(D)
                x_trial = np.clip(x + step, -5, 5)
                fx_trial = f(x_trial)

                n_trials += 1

                if fx_trial < fx:
                    # Success: increase sigma for dimensions that moved in the right direction
                    for d in range(D):
                        if step[d] != 0:
                            successes[d] += 1
                            # Increase sigma if the step was successful
                            sigmas[d] *= 1.25
                    x = x_trial
                    fx = fx_trial
                    if fx < best_f:
                        best_f = fx
                        best_x = np.array(x, dtype=float)
                else:
                    # Failure: decrease sigma for all dimensions
                    sigmas *= 0.8

                # Apply 1/5 rule: if success rate is below threshold, decrease all sigmas
                if n_trials >= 5:
                    success_rate = successes.sum() / (n_trials * D)
                    if success_rate < success_threshold:
                        sigmas *= 0.9
                    elif success_rate > 0.5:
                        sigmas *= 1.1
                    # Reset counters
                    successes[:] = 0
                    n_trials = 0

                # Convergence check
                if np.max(sigmas) < 1e-6:
                    break

        # Phase 2: Local polishing from the best point found
        if best_x is not None and calls[0] < B and time.time() < deadline:
            x = np.array(best_x, dtype=float)
            fx = best_f
            sigma = 0.1
            max_polish_iters = max(20, B - calls[0])

            for _ in range(max_polish_iters):
                if time.time() >= deadline or calls[0] >= B:
                    break

                step = sigma * rng.standard_normal(D)
                x_trial = np.clip(x + step, -5, 5)
                fx_trial = f(x_trial)

                if fx_trial < fx:
                    x = x_trial
                    fx = fx_trial
                    if fx < best_f:
                        best_f = fx
                        best_x = np.array(x, dtype=float)
                    sigma *= 1.1
                else:
                    sigma *= 0.8

                if sigma < 1e-7:
                    break

        if best_x is None:
            best_x = np.zeros(D)
            best_f = f(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
