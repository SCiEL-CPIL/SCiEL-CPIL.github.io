"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Hooke-Jeeves pattern search with adaptive step sizes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Initial step sizes: start large to explore, then adapt
        step = np.full(self.dim, 2.0)
        min_step = 1e-6
        max_step = 10.0
        expand_factor = 2.0
        shrink_factor = 0.5

        calls_used = 1
        remaining = self.budget - 1

        while remaining > 0:
            # Check deadline
            if time.time() > deadline:
                break

            improved = False
            for i in range(self.dim):
                if remaining <= 0 or time.time() > deadline:
                    break

                # Try positive direction
                x_trial = x.copy()
                x_trial[i] += step[i]
                x_trial = np.clip(x_trial, -5, 5)

                # If no change due to clipping, skip
                if np.array_equal(x_trial, x):
                    continue

                fy = func(x_trial)
                calls_used += 1
                remaining -= 1

                if fy < fx:
                    x = x_trial
                    fx = fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    improved = True
                    # Expand step in this direction
                    step[i] = min(step[i] * expand_factor, max_step)
                    # Continue exploring this coordinate (pattern search)
                    # Try to extend further in the same direction
                    while remaining > 0 and time.time() <= deadline:
                        x_trial2 = x.copy()
                        x_trial2[i] += step[i]
                        x_trial2 = np.clip(x_trial2, -5, 5)
                        if np.array_equal(x_trial2, x):
                            break
                        fy2 = func(x_trial2)
                        calls_used += 1
                        remaining -= 1
                        if fy2 < fx:
                            x = x_trial2
                            fx = fy2
                            if fx < best_fx:
                                best_fx = fx
                                best_x = x.copy()
                        else:
                            break
                else:
                    # Try negative direction
                    x_trial_neg = x.copy()
                    x_trial_neg[i] -= step[i]
                    x_trial_neg = np.clip(x_trial_neg, -5, 5)

                    if np.array_equal(x_trial_neg, x):
                        # Can't move in either direction, shrink step
                        step[i] = max(step[i] * shrink_factor, min_step)
                        continue

                    fy_neg = func(x_trial_neg)
                    calls_used += 1
                    remaining -= 1

                    if fy_neg < fx:
                        x = x_trial_neg
                        fx = fy_neg
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        improved = True
                        # Expand step in this direction
                        step[i] = min(step[i] * expand_factor, max_step)
                        # Continue exploring in the same direction
                        while remaining > 0 and time.time() <= deadline:
                            x_trial2 = x.copy()
                            x_trial2[i] -= step[i]
                            x_trial2 = np.clip(x_trial2, -5, 5)
                            if np.array_equal(x_trial2, x):
                                break
                            fy2 = func(x_trial2)
                            calls_used += 1
                            remaining -= 1
                            if fy2 < fx:
                                x = x_trial2
                                fx = fy2
                                if fx < best_fx:
                                    best_fx = fx
                                    best_x = x.copy()
                            else:
                                break
                    else:
                        # No improvement in either direction, shrink step
                        step[i] = max(step[i] * shrink_factor, min_step)

            # If no coordinate improved, we may be at a local minimum
            # Try a random restart from a different point if budget allows
            if not improved:
                if remaining > 50 and time.time() < deadline:
                    # Random restart
                    rng = np.random.default_rng(calls_used)
                    x_new = rng.uniform(-4, 4, self.dim)
                    fx_new = func(x_new)
                    calls_used += 1
                    remaining -= 1
                    if fx_new < best_fx:
                        best_fx = fx_new
                        best_x = x_new.copy()
                        x = x_new
                        fx = fx_new
                        step = np.full(self.dim, 2.0)
                    else:
                        # Keep current best, try smaller steps
                        step = np.full(self.dim, 0.5)
                else:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
