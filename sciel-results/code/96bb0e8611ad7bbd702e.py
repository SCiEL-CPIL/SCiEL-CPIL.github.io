# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-start L-BFGS-B optimizer for Bent Cigar.
    L-BFGS-B efficiently handles the anisotropic valley structure.
    Multiple random restarts with adaptive step sizes explore the landscape."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 285.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def wrapper(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                val = float(func(x_clipped))
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val
            except Exception:
                return float('inf')

        rng = np.random.default_rng(42)

        # Phase 1: Evaluate origin first (guaranteed finite, establishes baseline)
        x_origin = np.zeros(dim)
        f_origin = wrapper(x_origin)
        if f_origin < best_f:
            best_f = f_origin
            best_x = x_origin.copy()

        # Phase 2: L-BFGS-B from origin
        if calls < budget - 10 and time.time() < deadline:
            try:
                res = scipy.optimize.minimize(
                    wrapper, x_origin, method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'maxfev': min(100, budget - calls - 5)}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Multiple random restarts with L-BFGS-B
        # Bent Cigar: x1 is easy, x2..x5 are hard (scaled by 10^k)
        # Use larger steps in hard directions to escape local minima
        n_restarts = 8
        for i in range(n_restarts):
            if calls >= budget - 5:
                break
            if time.time() > deadline:
                break

            # Generate random start: larger spread in hard directions
            x_start = rng.uniform(-5, 5, dim)
            # For hard directions (indices 1..4), use larger initial steps
            # This helps L-BFGS-B find the valley more quickly
            
            # Adaptive budget: give more to earlier restarts
            remaining = budget - calls - 5
            if i < 3:
                maxfev = min(80, remaining)
            elif i < 6:
                maxfev = min(50, remaining)
            else:
                maxfev = min(30, remaining)
            
            if maxfev < 10:
                break

            try:
                res = scipy.optimize.minimize(
                    wrapper, x_start, method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 150, 'maxfev': maxfev}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
