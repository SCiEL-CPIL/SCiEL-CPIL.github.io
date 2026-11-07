"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, L_BFGS_B
import time


class Optimizer:
    """Multi-start L-BFGS-B with Nelder-Mead fallback and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = 320.0  # seconds, safety margin before 330s kill

    def __call__(self, func):
        start_time = time.time()
        dim = self.dim
        budget = self.budget

        # State tracking
        best_x = None
        best_f = np.inf
        call_count = 0
        func_calls = []  # list of (x, f) for potential reuse

        def make_eval(x):
            nonlocal call_count, best_x, best_f
            call_count += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            func_calls.append((x_clipped.copy(), f))
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Phase 1: Multi-start with diverse initializations
        # Use deterministic pseudo-random starts based on index
        rng = np.random.default_rng(42)

        # Strategy: Try several starting points, each with a limited budget
        # For dim=5, Rosenbrock has a known structure. The global min of standard
        # Rosenbrock is at (1,1,...,1), but here it's shifted/rotated.
        # We'll use a mix of center, random, and coordinate-wise starts.

        n_starts = 15  # Number of multi-start attempts
        calls_per_start = max(1, (budget - 10) // n_starts)  # Reserve 10 for polish

        for i in range(n_starts):
            if time.time() > self.deadline:
                break

            # Generate starting point
            if i == 0:
                x0 = np.zeros(dim)
            elif i == 1:
                x0 = np.ones(dim) * 3.0
            elif i == 2:
                x0 = np.ones(dim) * -3.0
            elif i == 3:
                x0 = np.ones(dim) * 1.0
            elif i == 4:
                x0 = np.ones(dim) * -1.0
            else:
                # Random starts with varying spread
                spread = 4.0 * (1.0 - i / (n_starts + 1))
                x0 = rng.uniform(-spread, spread, dim)

            x0 = np.clip(x0, -5.0, 5.0)

            # Evaluate starting point
            f0 = make_eval(x0)

            # Run L-BFGS-B with limited iterations
            try:
                res = minimize(
                    make_eval,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': calls_per_start, 'maxls': 20}
                )
            except Exception:
                pass

            if time.time() > self.deadline:
                break

        # Phase 2: Nelder-Mead polish on the best point found
        if best_x is not None and call_count < budget - 5:
            try:
                nm_res = minimize(
                    make_eval,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': min(50, budget - call_count - 5),
                             'xatol': 1e-10, 'fatol': 1e-12}
                )
            except Exception:
                pass

        # Phase 3: Final coordinate-wise line search polish
        if best_x is not None and call_count < budget - 2:
            x = best_x.copy()
            f = best_f
            # Simple coordinate descent with small steps
            for _ in range(10):
                if time.time() > self.deadline or call_count >= budget - 1:
                    break
                improved = False
                for d in range(dim):
                    for delta in [1e-3, 1e-4, 1e-5, -1e-3, -1e-4, -1e-5]:
                        if call_count >= budget - 1:
                            break
                        x_try = x.copy()
                        x_try[d] += delta
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = make_eval(x_try)
                        if f_try < f - 1e-12:
                            x = x_try
                            f = f_try
                            improved = True
                            break
                if not improved:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
