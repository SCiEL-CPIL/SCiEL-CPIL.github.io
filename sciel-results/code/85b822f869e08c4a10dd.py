"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart L-BFGS-B with adaptive exploration for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = np.zeros(self.dim)
        best_f = float('inf')
        call_count = 0
        calls_made = 0

        def tracked_func(x):
            nonlocal call_count, calls_made
            call_count += 1
            calls_made += 1
            if calls_made >= self.budget:
                return float('inf')
            return func(np.clip(x, -5, 5))

        # Phase 1: Multi-restart L-BFGS-B
        # Use different random starting points
        n_restarts = 15
        restarts_done = 0

        for restart in range(n_restarts):
            if time.time() > deadline - 30:  # leave time for final refinement
                break
            if calls_made >= self.budget:
                break

            # Generate diverse starting points
            if restart == 0:
                x0 = np.zeros(self.dim)
            elif restart == 1:
                x0 = rng.uniform(-2, 2, self.dim)
            elif restart == 2:
                x0 = rng.uniform(-4, 4, self.dim)
            else:
                # Mix of random and structured starts
                if restart % 3 == 0:
                    x0 = rng.uniform(-5, 5, self.dim)
                elif restart % 3 == 1:
                    # Sparse random: mostly zero with a few random coords
                    x0 = np.zeros(self.dim)
                    n_nonzero = rng.integers(1, self.dim)
                    idx = rng.choice(self.dim, n_nonzero, replace=False)
                    x0[idx] = rng.uniform(-4, 4, n_nonzero)
                else:
                    # Radial sampling
                    direction = rng.standard_normal(self.dim)
                    direction /= np.linalg.norm(direction)
                    radius = rng.uniform(0, 4)
                    x0 = direction * radius

            x0 = np.clip(x0, -5, 5)

            try:
                result = minimize(
                    tracked_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'maxfun': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                continue

            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()

            if calls_made >= self.budget:
                break

            # Early exit if we're doing very well
            if best_f < 1e-6:
                break

        # Phase 2: Local refinement with Nelder-Mead around best point
        if calls_made < self.budget and time.time() < deadline - 10:
            try:
                result = minimize(
                    tracked_func,
                    best_x,
                    method='Nelder-Mead',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'maxfev': 50, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 3: If budget remains, do a final coordinate descent polish
        if calls_made < self.budget and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            step = 0.1
            for coord in range(self.dim):
                for direction in [-1, 1]:
                    if calls_made >= self.budget:
                        break
                    x_trial = x.copy()
                    x_trial[coord] = np.clip(x_trial[coord] + direction * step, -5, 5)
                    f_trial = tracked_func(x_trial)
                    if f_trial < fx:
                        x = x_trial
                        fx = f_trial
                        step *= 1.2
                    else:
                        step *= 0.7

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
