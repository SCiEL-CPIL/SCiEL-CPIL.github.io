"""Optimized BBOB continuous optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with multi-start and time-bounded execution."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        calls_made = 0
        best_x = None
        best_f = np.inf

        # Count calls safely
        def safe_call(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return np.inf
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if x_clipped.shape != (self.dim,):
                x_clipped = x_clipped.reshape(self.dim)
            val = float(func(x_clipped))
            calls_made += 1
            return val

        # Update best
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Multi-start CMA-ES
        n_starts = 5
        per_start_budget = self.budget // n_starts

        for s in range(n_starts):
            if time.time() > deadline:
                break
            if calls_made >= self.budget:
                break

            # Different starting points
            if s == 0:
                x0 = np.zeros(self.dim)
            elif s == 1:
                x0 = np.full(self.dim, 2.0)
            elif s == 2:
                x0 = np.full(self.dim, -2.0)
            elif s == 3:
                rng = np.random.default_rng(42 + s)
                x0 = rng.uniform(-4, 4, self.dim)
            else:
                rng = np.random.default_rng(100 + s)
                x0 = rng.uniform(-4, 4, self.dim)

            # Check time before starting
            if time.time() > deadline:
                break

            # Try CMA-ES via scipy
            try:
                result = minimize(
                    safe_call,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': per_start_budget,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                    }
                )
                if result.fun < best_f:
                    update_best(result.x, result.fun)
            except Exception:
                pass

            # Also try a few random perturbations from best
            if best_x is not None and time.time() < deadline and calls_made < self.budget:
                rng = np.random.default_rng(200 + s)
                for _ in range(min(10, self.budget - calls_made)):
                    if time.time() > deadline:
                        break
                    if calls_made >= self.budget:
                        break
                    pert = best_x + rng.normal(0, 0.5, self.dim)
                    f_pert = safe_call(pert)
                    update_best(pert, f_pert)

        # Final local refinement with finer steps if time allows
        if best_x is not None and time.time() < deadline and calls_made < self.budget:
            # Coordinate descent refinement
            for coord_pass in range(3):
                if time.time() > deadline or calls_made >= self.budget:
                    break
                for d in range(self.dim):
                    if time.time() > deadline or calls_made >= self.budget:
                        break
                    # Try small steps in each direction
                    for step in [0.1, -0.1, 0.01, -0.01, 0.001, -0.001]:
                        if calls_made >= self.budget or time.time() > deadline:
                            break
                        trial = best_x.copy()
                        trial[d] += step
                        f_trial = safe_call(trial)
                        update_best(trial, f_trial)

        return best_x, best_f
# EVOLVE-BLOCK-END
