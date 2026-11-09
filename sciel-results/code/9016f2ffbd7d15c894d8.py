"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid Nelder-Mead + L-BFGS-B optimizer for Rosenbrock with hidden shifts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

        n = self.dim
        total_budget = self.budget
        calls = 0
        best_f = np.inf
        best_x = np.zeros(n)
        history = []

        def safe_func(x):
            nonlocal calls, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if calls >= total_budget:
                raise StopIteration("Budget exceeded")

            x = np.asarray(x, dtype=float).reshape(n)
            x_clipped = np.clip(x, -5, 5)
            if not np.all(np.isfinite(x_clipped)):
                return 1e10

            f = func(x_clipped)
            calls += 1

            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()

            return f

        rng = np.random.default_rng(42)

        # Phase 1: Structured exploration - cover the likely minimizer region [-4,4]^5
        # Use a mix of deterministic and random points, biased toward center
        starts = []
        # Center and near-center points
        starts.append(np.zeros(n))
        for scale in [0.5, 1.0, 2.0, 3.0]:
            for corner in [(1,1,1,1,1), (-1,-1,-1,-1,-1), (1,-1,1,-1,1), (-1,1,-1,1,-1)]:
                starts.append(np.array(corner, dtype=float) * scale)
        # Random points with decreasing radius
        for radius in [4.0, 3.0, 2.0, 1.0, 0.5]:
            for _ in range(3):
                starts.append(rng.normal(0, radius, size=n))

        n_explore = min(len(starts), max(20, int(total_budget * 0.25)))
        starts = starts[:n_explore]

        for x0 in starts:
            if calls >= total_budget or time.time() > deadline:
                break
            try:
                f0 = safe_func(x0)
                history.append((f0, x0.copy()))
            except (StopIteration, TimeoutError):
                break

        if not history:
            return best_x, best_f

        history.sort(key=lambda item: item[0])

        # Phase 2: Nelder-Mead refinement on top candidates
        # Nelder-Mead is derivative-free and handles the Rosenbrock valley well
        # without wasting calls on line searches
        top_k = min(8, len(history))
        nm_used = 0

        for i in range(top_k):
            if calls >= total_budget or time.time() > deadline:
                break

            x_start = history[i][1].copy()

            # If we've found a very good solution, do a final L-BFGS-B polish
            if best_f < 1.0 and nm_used >= 2:
                break

            try:
                res = minimize(
                    safe_func,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 300,
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'adaptive': True,
                        'disp': False
                    }
                )
                nm_used += 1
            except (StopIteration, TimeoutError):
                break
            except Exception:
                continue

        # Phase 3: Final L-BFGS-B polish if budget remains
        if calls < total_budget and time.time() < deadline and best_f < 100.0:
            try:
                res = minimize(
                    safe_func,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-15,
                        'gtol': 1e-12,
                        'disp': False,
                        'maxls': 30
                    }
                )
            except (StopIteration, TimeoutError):
                pass
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
