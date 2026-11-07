"""Rewritten optimizer using Nelder-Mead with stratified global exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Stratified Nelder-Mead optimizer for BBOB f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 305.0

        def _tl():
            return deadline - time.time()

        rng = np.random.default_rng(12345)
        calls = 0
        best_f = np.inf
        best_x = None
        d = self.dim
        B = self.budget

        def ev(x):
            nonlocal calls, best_f, best_x
            if calls >= B:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            calls += 1
            fx = float(func(x))
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Stratified global sampling ---
        # Divide each dimension into strata for more uniform coverage
        n_strata = 4
        edges = np.linspace(-5.0, 5.0, n_strata + 1)
        stratified_pts = []
        for idx in range(n_strata ** d):
            if calls >= B or _tl() < 30:
                break
            coords = []
            rem = idx
            for dim_i in range(d):
                s = rem % n_strata
                rem //= n_strata
                lo, hi = edges[s], edges[s + 1]
                coords.append(rng.uniform(lo, hi))
            stratified_pts.append(np.array(coords))

        # Evaluate stratified points (shuffle for diversity in early progress)
        order = rng.permutation(len(stratified_pts))
        for i in order:
            if calls >= B or _tl() < 30:
                break
            ev(stratified_pts[i])

        # --- Phase 2: Nelder-Mead local refinement from top starts ---
        # Build a list of starting points: best + diverse perturbations
        starts = []
        if best_x is not None:
            starts.append(best_x.copy())

        # Add points at various scales around best
        for scale in [0.3, 0.7, 1.5, 3.0]:
            if calls >= B or _tl() < 15:
                break
            for _ in range(3):
                p = best_x + scale * rng.standard_normal(d)
                p = np.clip(p, -5.0, 5.0)
                starts.append(p)

        # Also add some fully random starts
        for _ in range(5):
            if calls >= B or _tl() < 15:
                break
            starts.append(rng.uniform(-5, 5, size=d))

        # Evaluate all starts, sort by fitness
        start_fits = []
        for s in starts:
            if calls >= B or _tl() < 10:
                break
            f = ev(s)
            start_fits.append((f, s.copy()))
        start_fits.sort(key=lambda t: t[0])

        # Run Nelder-Mead from top starts
        nm_maxfun = 40
        for fi, sx in start_fits:
            if calls >= B or _tl() < 10:
                break
            try:
                res = minimize(
                    lambda v: ev(v),
                    sx,
                    method='Nelder-Mead',
                    options={'maxiter': 80, 'maxfev': nm_maxfun,
                             'xatol': 0.01, 'fatol': 1e-8,
                             'adaptive': True}
                )
            except Exception:
                pass

        # --- Phase 3: Momentum hill-climbing refinement ---
        # If we still have budget, do gradient-free momentum steps
        if best_x is not None and calls < B and _tl() > 5:
            x = best_x.copy()
            f_cur = best_f
            step_size = 0.5
            momentum = np.zeros(d)
            lr = 0.3

            for step_i in range(200):
                if calls >= B or _tl() < 3:
                    break
                # Try step in random direction with momentum
                direction = rng.standard_normal(d)
                direction /= (np.linalg.norm(direction) + 1e-12)
                momentum = lr * momentum + direction
                new_x = np.clip(x + step_size * momentum, -5.0, 5.0)
                f_new = ev(new_x)

                if f_new < f_cur:
                    x = new_x
                    f_cur = f_new
                    step_size = min(step_size * 1.1, 3.0)
                else:
                    step_size = max(step_size * 0.7, 0.05)
                    if step_size < 0.05:
                        step_size = rng.uniform(0.5, 2.0)
                        momentum = np.zeros(d)

        # --- Phase 4: Final scattered random sampling ---
        while calls < B and _tl() > 2:
            if best_x is not None:
                s = rng.uniform(0.2, 4.0)
                x_try = np.clip(best_x + s * rng.standard_normal(d), -5.0, 5.0)
            else:
                x_try = rng.uniform(-5, 5, size=d)
            ev(x_try)

        return best_x, best_f
# EVOLVE-BLOCK-END
