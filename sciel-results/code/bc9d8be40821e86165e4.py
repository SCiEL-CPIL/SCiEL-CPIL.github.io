"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hierarchical coarse search + per-dimension adaptive (1+1)-ES with periodic FD gradient polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget

        # Global best tracking
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            return float(func(x))

        def time_ok():
            return time.time() < deadline

        # ===== Phase 1: Coarse multi-scale random walk for early progress =====
        # Use ~15% of budget for broad exploration with geometric step reduction
        coarse_budget = max(15, total_budget // 7)
        coarse_levels = 4
        current_scale = 4.0

        for scale_level in range(coarse_levels):
            scale_step = current_scale * (0.5 ** scale_level)
            n_evals = max(3, coarse_budget // coarse_levels)

            for _ in range(n_evals):
                if calls_made >= total_budget or not time_ok():
                    break
                if best_x is None:
                    x = rng.uniform(-5, 5, dim)
                else:
                    x = best_x + scale_step * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)

                fx = safe_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            current_scale *= 0.5

        if best_x is None:
            x = rng.uniform(-5, 5, dim)
            best_f = safe_func(x)
            best_x = x.copy()

        # ===== Phase 2: Per-dimension adaptive (1+1)-ES with periodic FD gradient polish =====
        # Structural idea from inspiration 1: per-dimension sigma with windowed success tracking
        # Refinement machinery: periodic finite-difference gradient polish

        x = best_x.copy()
        fx = best_f
        sigma = np.full(dim, 1.0)  # Per-dimension step sizes

        # Windowed success tracking per dimension
        window = 10
        recent_successes = [0] * dim
        recent_trials = [0] * dim

        polish_interval = 40  # Every 40 iterations, do FD gradient polish
        last_polish_iter = 0

        for iteration in range(total_budget - calls_made):
            if not time_ok() or calls_made >= total_budget:
                break

            # Periodic FD gradient polish
            if iteration - last_polish_iter >= polish_interval and calls_made < total_budget - (2 * dim + 2):
                eps = 1e-4
                grad = np.zeros(dim)
                for i in range(dim):
                    if not time_ok() or calls_made >= total_budget - 2:
                        break
                    xp = x.copy()
                    xm = x.copy()
                    xp[i] += eps
                    xm[i] -= eps
                    xp = np.clip(xp, -5, 5)
                    xm = np.clip(xm, -5, 5)
                    fp = safe_func(xp)
                    fm = safe_func(xm)
                    grad[i] = (fp - fm) / (2 * eps)

                # Gradient descent step with adaptive step size
                gnorm = np.linalg.norm(grad)
                if gnorm > 1e-10:
                    step_size = min(0.5, gnorm * 0.1)
                    x_new = np.clip(x - step_size * grad, -5, 5)
                    fx_new = safe_func(x_new)
                    if fx_new < fx:
                        x = x_new
                        fx = fx_new
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    # Reduce sigma after polish (we've moved closer to optimum)
                    sigma *= 0.85
                    sigma = np.clip(sigma, 0.005, 10.0)

                last_polish_iter = iteration

            if not time_ok() or calls_made >= total_budget:
                break

            # Per-dimension adaptive (1+1)-ES step
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = safe_func(y)

            if fy < fx:
                # Success
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                # Track per-dimension success
                for i in range(dim):
                    recent_successes[i] += 1
                    recent_trials[i] += 1
                    if recent_trials[i] >= window:
                        rate = recent_successes[i] / recent_trials[i]
                        if rate > 0.5:
                            sigma[i] *= 1.2
                        elif rate < 0.5:
                            sigma[i] *= 0.8
                        recent_successes[i] = 0
                        recent_trials[i] = 0
                sigma = np.clip(sigma, 0.005, 10.0)
            else:
                # Failure
                for i in range(dim):
                    recent_trials[i] += 1
                    if recent_trials[i] >= window:
                        rate = recent_successes[i] / recent_trials[i]
                        if rate < 0.3:
                            sigma[i] *= 0.7
                        elif rate < 0.5:
                            sigma[i] *= 0.9
                        recent_successes[i] = 0
                        recent_trials[i] = 0
                sigma = np.clip(sigma, 0.005, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
