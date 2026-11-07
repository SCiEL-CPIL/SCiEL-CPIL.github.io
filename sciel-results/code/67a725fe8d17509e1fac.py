"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES-lite with per-dimension adaptive step sizes and periodic gradient polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Initialize position and per-dimension step sizes
        x = rng.uniform(-4, 4, d)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # Per-dimension step sizes
        sigma = np.full(d, 1.0)

        # Success counters per dimension (recent window)
        window_size = 10
        successes = np.zeros((d, window_size), dtype=int)
        succ_idx = 0

        calls_made = 1
        polish_interval = 50
        calls_since_polish = 0

        # Finite-difference gradient polish helper
        def polish(x_cur, f_cur, remaining_budget):
            nonlocal best_x, best_f, calls_made
            if remaining_budget < 15:
                return x_cur, f_cur
            eps = 1e-4
            x_work = x_cur.copy()
            f_work = f_cur
            # Do a few gradient descent steps with estimated gradient
            for _ in range(5):
                if calls_made >= budget:
                    break
                grad = np.zeros(d)
                for i in range(d):
                    if calls_made >= budget:
                        break
                    xp = x_work.copy()
                    xp[i] += eps
                    fp = func(np.clip(xp, -5, 5))
                    calls_made += 1
                    xm = x_work.copy()
                    xm[i] -= eps
                    fm = func(np.clip(xm, -5, 5))
                    calls_made += 1
                    grad[i] = (fp - fm) / (2 * eps)
                # Normalize gradient step
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-10:
                    break
                step = grad / gnorm
                # Try a few step sizes
                found_improvement = False
                for lr in [0.5, 0.2, 0.1, 0.05]:
                    if calls_made >= budget:
                        break
                    x_try = np.clip(x_work - lr * step, -5, 5)
                    f_try = func(x_try)
                    calls_made += 1
                    if f_try < f_work:
                        x_work = x_try
                        f_work = f_try
                        found_improvement = True
                        break
                if not found_improvement:
                    break
            if f_work < best_f:
                best_x = x_work.copy()
                best_f = f_work
            return x_work, f_work

        while calls_made < budget:
            # Generate candidate
            y = x + sigma * rng.standard_normal(d)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1
            calls_since_polish += 1

            if fy < fx:
                # Accept move
                x, fx = y, fy
                if fx < best_f:
                    best_x = x.copy()
                    best_f = fx
                # Update per-dimension success
                for i in range(d):
                    successes[i, succ_idx] = 1
            else:
                for i in range(d):
                    successes[i, succ_idx] = 0

            succ_idx = (succ_idx + 1) % window_size

            # Update step sizes based on success rate
            for i in range(d):
                rate = np.mean(successes[i])
                if rate > 0.5:
                    sigma[i] *= 1.1
                elif rate < 0.5:
                    sigma[i] *= 0.9
                # Clamp step sizes
                sigma[i] = np.clip(sigma[i], 0.01, 5.0)

            # Periodic polish phase
            if calls_since_polish >= polish_interval and calls_made < budget:
                calls_since_polish = 0
                x, fx = polish(x, fx, budget - calls_made)
                # Re-center sigma around current position
                # Keep sigma as is, but ensure we're tracking best

        # Final polish if budget allows
        if calls_made < budget:
            polish(best_x, best_f, budget - calls_made)

        return best_x, best_f
# EVOLVE-BLOCK-END
