"""Hybrid Powell/L-BFGS-B optimizer for Bent Cigar (f12)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid Powell (search) + L-BFGS-B (polish) for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        calls = [0]
        best_x = None
        best_f = np.inf
        deadline = time.time() + 280  # safety margin before 330s kill

        def F(x):
            calls[0] += 1
            if calls[0] > budget:
                return 1e10
            val = func(np.asarray(x, dtype=float))
            return val

        bounds = [(-5.0, 5.0)] * d
        rng = np.random.default_rng(42)

        # Powell adapts directions, good for ill-conditioned valleys.
        # L-BFGS-B polishes the local minimum.
        n_restarts = 5

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if calls[0] >= budget - 20:
                break

            # Generate diverse starting points
            if r == 0:
                x0 = np.zeros(d)
            elif r == 1:
                x0 = rng.uniform(-2.0, 2.0, d)
            elif r == 2:
                x0 = rng.uniform(-4.0, 4.0, d)
            else:
                x0 = rng.uniform(-5.0, 5.0, d)

            # Clip to bounds
            x0 = np.clip(x0, -5.0, 5.0)

            # Evaluate starting point
            f0 = F(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            if calls[0] >= budget - 20:
                break

            # Phase 1: Powell for global search / direction alignment
            # Powell is derivative-free and adapts axes.
            # Limit function calls for this phase to leave room for polish.
            powell_budget = min(40, budget - calls[0] - 10)
            
            try:
                res_powell = minimize(
                    F, x0, method='Powell',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'maxfev': powell_budget,
                        'xtol': 1e-6,
                        'ftol': 1e-6,
                    }
                )
            except Exception:
                pass
            else:
                if res_powell.fun < best_f:
                    best_f = res_powell.fun
                    best_x = np.clip(res_powell.x, -5.0, 5.0)

            if calls[0] >= budget - 10:
                break

            # Phase 2: L-BFGS-B polish
            # Start from the best point found by Powell (or initial if Powell failed)
            # Use a small budget for polishing to ensure we stay within limits.
            polish_budget = min(20, budget - calls[0] - 5)
            
            if polish_budget > 0:
                try:
                    res_lfgs = minimize(
                        F, best_x, method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 50,
                            'maxfun': polish_budget,
                            'gtol': 1e-10,
                            'ftol': 1e-12,
                        }
                    )
                except Exception:
                    pass
                else:
                    if res_lfgs.fun < best_f:
                        best_f = res_lfgs.fun
                        best_x = np.clip(res_lfgs.x, -5.0, 5.0)

        if best_x is None:
            best_x = np.zeros(d)
            best_f = F(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
