"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-lite with per-dimension adaptive step sizes, periodic gradient polishing,
    and stagnation detection with local restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        t_start = time.time()
        deadline = t_start + 280.0  # 20s safety margin before 300s kill

        # State
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Per-dimension step sizes
        sigma = np.full(dim, 0.5)

        # Success tracking per dimension
        window = 20
        successes = [np.zeros(window) for _ in range(dim)]
        trial_count = 0

        # Stagnation tracking
        stagnation_count = 0
        best_fx_at_stagnation = fx

        def time_ok():
            return time.time() < deadline and calls < budget

        def estimate_gradient(x0, f0, step=1e-4):
            """Finite-difference gradient estimation."""
            nonlocal calls
            grad = np.zeros(dim)
            for i in range(dim):
                if not time_ok():
                    break
                xp = x0.copy()
                xm = x0.copy()
                xp[i] += step
                xm[i] -= step
                xp = np.clip(xp, -5, 5)
                xm = np.clip(xm, -5, 5)
                fpx = func(xp)
                calls += 1
                if not time_ok():
                    break
                fmx = func(xm)
                calls += 1
                grad[i] = (fpx - fmx) / (2 * step)
            return grad

        def polish(x0, f0, max_iter=30):
            """Gradient-based polishing with adaptive step size."""
            nonlocal calls, best_x, best_fx
            x = x0.copy()
            fx = f0
            lr = 0.1
            for it in range(max_iter):
                if not time_ok():
                    break
                grad = estimate_gradient(x, fx)
                if not time_ok():
                    break
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-10:
                    break
                # Line search: try step, accept if improvement
                direction = -grad / gnorm
                step_size = lr
                for _ in range(5):
                    if not time_ok():
                        break
                    xn = np.clip(x + step_size * direction, -5, 5)
                    fn = func(xn)
                    calls += 1
                    if fn < fx:
                        x = xn
                        fx = fn
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        break
                    else:
                        step_size *= 0.5
                else:
                    # No improvement, reduce learning rate
                    lr *= 0.5
                    if lr < 1e-8:
                        break
            return x, fx

        def local_restart(x0, f0, n_restarts=5):
            """Try several random perturbations around x0 to escape local minima."""
            nonlocal calls, best_x, best_fx
            for _ in range(n_restarts):
                if not time_ok():
                    break
                perturb = rng.uniform(-1, 1, dim) * 0.5
                xr = np.clip(x0 + perturb, -5, 5)
                fr = func(xr)
                calls += 1
                if fr < best_fx:
                    best_fx = fr
                    best_x = xr.copy()
            return best_x, best_fx

        # Main loop
        while time_ok():
            # CMA-ES-lite step: propose one move
            y = x.copy()
            for i in range(dim):
                y[i] += sigma[i] * rng.standard_normal()
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            # Update per-dimension success and step size
            for i in range(dim):
                if y[i] != x[i]:
                    success_i = 1.0 if fy < fx else 0.0
                    successes[i][trial_count % window] = success_i
                    if trial_count >= window:
                        rate = np.mean(successes[i])
                        if rate > 0.5:
                            sigma[i] = min(sigma[i] * 1.1, 5.0)
                        elif rate < 0.3:
                            sigma[i] = max(sigma[i] * 0.8, 0.01)
            trial_count += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            else:
                stagnation_count += 1

            # Periodic polishing every 50 calls
            if calls % 50 == 0 and time_ok():
                x, fx = polish(best_x, best_fx, max_iter=15)
                # Reset stagnation if polish helped
                if fx < best_fx_at_stagnation:
                    stagnation_count = 0
                    best_fx_at_stagnation = fx

            # Stagnation detection: if no improvement for 80 calls, do local restart
            if stagnation_count >= 80 and time_ok():
                best_x, best_fx = local_restart(best_x, best_fx, n_restarts=10)
                x = best_x.copy()
                fx = best_fx
                sigma[:] = np.maximum(sigma * 0.5, 0.05)
                stagnation_count = 0
                best_fx_at_stagnation = fx

        # Final polish if time permits
        if time_ok():
            polish(best_x, best_fx, max_iter=20)

        return best_x, best_fx
# EVOLVE-BLOCK-END
