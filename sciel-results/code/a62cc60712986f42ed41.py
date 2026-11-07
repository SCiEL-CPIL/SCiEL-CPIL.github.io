"""Rewritten optimizer for BBOB f22 (Gallagher 21 peaks)."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Hybrid: random restarts + L-BFGS-B local refinement with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 280  # 280s safety margin before 300s kill
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def bounded_func(x):
            nonlocal calls_made
            x_clipped = np.clip(x, -5.0, 5.0)
            calls_made += 1
            return func(x_clipped)

        # Phase 1: Broad random sampling to find diverse basins
        n_random = 80
        candidates = []
        for _ in range(n_random):
            x = rng.uniform(-5.0, 5.0, self.dim)
            fx = bounded_func(x)
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort by fitness, keep top diverse candidates
        candidates.sort(key=lambda t: t[0])
        top_candidates = []
        min_dist = 1.0  # minimum distance between kept candidates
        for fx, x in candidates:
            if calls_made >= self.budget:
                break
            if time.time() > deadline - 10:
                break
            diverse = True
            for _, y in top_candidates:
                if np.linalg.norm(x - y) < min_dist:
                    diverse = False
                    break
            if diverse:
                top_candidates.append((fx, x))
            if len(top_candidates) >= 15:
                break

        # Phase 2: Local refinement with L-BFGS-B on each candidate
        for fx0, x0 in top_candidates:
            if calls_made >= self.budget:
                break
            if time.time() > deadline - 5:
                break

            # Try L-BFGS-B
            try:
                res = minimize(
                    bounded_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * self.dim,
                    options={'maxiter': 200, 'maxls': 20, 'maxfun': 50}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

            if calls_made >= self.budget:
                break
            if time.time() > deadline - 5:
                break

            # If L-BFGS-B didn't improve much, try Nelder-Mead as backup
            if len(top_candidates) <= 5:  # only for top few
                try:
                    res2 = minimize(
                        bounded_func,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8}
                    )
                    if res2.fun < best_f:
                        best_f = res2.fun
                        best_x = np.clip(res2.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        # Phase 3: If we still have budget, do more random restarts + refine
        extra_rounds = 5
        for _ in range(extra_rounds):
            if calls_made >= self.budget:
                break
            if time.time() > deadline - 5:
                break

            # Generate a few more random points
            for _ in range(20):
                if calls_made >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break
                x = rng.uniform(-5.0, 5.0, self.dim)
                fx = bounded_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            # Refine the current best with a fresh L-BFGS-B run
            if best_x is not None and calls_made < self.budget:
                try:
                    res = minimize(
                        bounded_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * self.dim,
                        options={'maxiter': 150, 'maxls': 20, 'maxfun': 30}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        return best_x, best_f
