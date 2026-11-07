"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, multi-start, and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 330s kill

        # Track best
        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Multi-start (1+1)-ES with adaptive sigma
        # Use 3 restarts, each with 100 budget
        n_restarts = 3
        per_restart = 100

        for r in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break

            # Start point: center for first, random for others
            if r == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4, 4, dim)

            fx = eval_func(x)
            sigma = 1.5

            # Success rate tracking for sigma adaptation
            successes = 0
            window = 10
            recent = [False] * window
            idx = 0

            for _ in range(per_restart):
                if calls >= budget or time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    recent[idx % window] = True
                else:
                    recent[idx % window] = False
                idx += 1

                # Adapt sigma based on success rate in recent window
                if idx >= window:
                    sr = sum(recent) / window
                    if sr > 0.4:
                        sigma *= 1.1
                    elif sr < 0.2:
                        sigma *= 0.9
                    else:
                        sigma *= 1.0

                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.01, 5.0)

        # Phase 2: Local refinement using coordinate search on best point
        if best_x is not None and calls < budget and time.time() < deadline:
            # Simple local search with decreasing step
            x_local = best_x.copy()
            f_local = best_f
            step = 0.5
            # Use remaining budget, but cap iterations to avoid infinite loops if step doesn't shrink fast enough
            max_iters = budget - calls
            for _ in range(max_iters):
                if calls >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    for sign in [1, -1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        y = x_local.copy()
                        y[d] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = eval_func(y)
                        if fy < f_local:
                            x_local, f_local = y, fy
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 0.01:
                        break
                else:
                    # If improved, we might want to keep step or increase slightly? 
                    # For Rastrigin, small steps are better. Keep step same or decrease slightly.
                    # Let's just keep the same step for a bit if improving.
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
