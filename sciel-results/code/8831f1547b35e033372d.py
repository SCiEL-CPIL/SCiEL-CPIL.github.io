"""Black-box continuous optimizer on the BBOB-style function f12 (Bent cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def time_left():
            return time.time() < deadline

        def make_step():
            nonlocal calls
            if not time_left():
                return None
            calls += 1
            return calls

        # Phase 1: Multi-start (1+1)-ES with restarts
        # We use a simple approach: run (1+1)-ES from several random starts,
        # each with a subset of the budget.

        # First, do a quick random search to get a reasonable starting point
        # and estimate the scale.
        n_init = 10
        init_xs = []
        init_fs = []
        for i in range(n_init):
            if not time_left():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            init_xs.append(x)
            init_fs.append(fx)

        if not init_xs:
            # Could not even make one call
            return np.zeros(dim), np.inf

        # Pick the best initial point
        best_idx = np.argmin(init_fs)
        x = init_xs[best_idx].copy()
        fx = init_fs[best_idx]
        best_x = x.copy()
        best_f = fx

        # Now run (1+1)-ES with restarts
        # Strategy: split remaining budget into several runs, each starting
        # from a different point (best so far + perturbation).
        remaining_budget = budget - calls
        if remaining_budget <= 0:
            return best_x, best_f

        # Use a simple (1+1)-ES with 1/5 rule for sigma adaptation
        # with periodic restarts to escape local minima.

        # Number of restarts: we'll do several segments
        n_restarts = 5
        seg_budget = max(1, remaining_budget // n_restarts)

        for restart in range(n_restarts):
            if not time_left():
                break

            if restart == 0:
                # Continue from current best
                x_cur = best_x.copy()
                f_cur = best_f
            else:
                # Restart from a perturbation of the best
                sigma_restart = 1.0
                x_cur = np.clip(best_x + sigma_restart * rng.standard_normal(dim), -5, 5)
                f_cur = func(x_cur)
                calls += 1
                if f_cur < best_f:
                    best_x = x_cur.copy()
                    best_f = f_cur

            sigma = 1.0
            # 1/5 rule parameters
            window = 10
            successes = 0
            step_count = 0

            for step in range(seg_budget):
                if not time_left():
                    break

                # Generate trial point
                y = np.clip(x_cur + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1

                successes += 1 if fy < f_cur else 0
                step_count += 1

                if fy < f_cur:
                    x_cur = y
                    f_cur = fy
                    if f_cur < best_f:
                        best_x = x_cur.copy()
                        best_f = f_cur

                # Adapt sigma every 'window' steps
                if step_count % window == 0:
                    success_rate = successes / window
                    if success_rate > 0.5:
                        sigma *= 1.3
                    elif success_rate < 0.1:
                        sigma *= 0.7
                    else:
                        sigma *= 1.0
                    successes = 0
                    step_count = 0

                # Clamp sigma
                sigma = np.clip(sigma, 1e-6, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
