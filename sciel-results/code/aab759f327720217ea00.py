"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid optimizer: structured initialization + multi-restart Powell + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        calls_made = 0
        dim = self.dim
        budget = self.budget

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Structured initialization to find good basins
        # Try origin first (strong prior for unshifted Rastrigin), then sample in [-4, 4]^5
        n_init = 100
        for i in range(n_init):
            if calls_made >= budget or time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4, 4, dim)
            counted_func(x)

        # Phase 2: Multi-restart Powell local refinement
        # Powell is derivative-free and robust for multimodal functions
        max_powell_runs = 5
        for run_idx in range(max_powell_runs):
            if calls_made >= budget or time.time() > deadline - 10:
                break

            if run_idx == 0:
                x_start = best_x.copy()
            else:
                # Perturb the best point to escape local minima
                sigma = 0.5 / (run_idx + 1)
                x_start = best_x + rng.normal(0, sigma, dim)
                x_start = clip(x_start)

            try:
                minimize(
                    counted_func,
                    x_start,
                    method='Powell',
                    options={'maxiter': 100, 'maxfev': 100}
                )
            except Exception:
                pass

        # Phase 3: Final polish with Nelder-Mead for high precision
        if calls_made < budget and time.time() < deadline - 5:
            try:
                minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
