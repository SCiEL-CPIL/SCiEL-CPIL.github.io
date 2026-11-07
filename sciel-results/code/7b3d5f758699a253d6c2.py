"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start + Nelder-Mead simplex + adaptive polish for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or not time_ok():
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                fx = best_f
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start exploration - sample diverse points
        n_initial = 10
        candidates = []
        for i in range(n_initial):
            if calls >= budget or not time_ok():
                break
            if i == 0:
                x = np.zeros(n)
            elif i == 1:
                x = rng.uniform(-4, 4, n)
            elif i == 2:
                x = rng.uniform(-2, 2, n)
            elif i == 3:
                x = rng.uniform(-5, 5, n)
            else:
                x = rng.uniform(-5, 5, n)
            fx = evaluate(x)
            if fx is not None:
                candidates.append((fx, x.copy()))

        if not candidates:
            x = rng.uniform(-5, 5, n)
            evaluate(x)
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])

        # Phase 2: Nelder-Mead on top-2 candidates
        top_k = min(2, len(candidates))
        remaining = budget - calls

        for rank in range(top_k):
            if calls >= budget or not time_ok():
                break

            f0, x0 = candidates[rank]
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            # Build simplex: best point + small perturbations
            simplex = [x0.copy()]
            for j in range(n):
                point = x0.copy()
                # Choose direction that stays in bounds
                if x0[j] < 4.0:
                    delta = 0.5
                else:
                    delta = -0.5
                point[j] += delta
                point = np.clip(point, -5, 5)
                simplex.append(point)

            fvals = [f0]
            for pt in simplex[1:]:
                if calls >= budget or not time_ok():
                    break
                fx = evaluate(pt)
                if fx is None:
                    break
                fvals.append(fx)

            if len(fvals) != len(simplex):
                continue

            max_nm_iter = max(0, (budget - calls) // max(1, top_k - rank))
            for _ in range(max_nm_iter):
                if calls >= budget or not time_ok():
                    break

                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if abs(fvals[-1] - fvals[0]) < 1e-10:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                fr = evaluate(xr)
                if fr is None:
                    break

                if fr < fvals[0]:
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    fe = evaluate(xe)
                    if fe is None:
                        simplex[-1], fvals[-1] = xr, fr
                        break
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                elif fr < fvals[-2]:
                    simplex[-1], fvals[-1] = xr, fr
                else:
                    if fr < fvals[-1]:
                        xc = np.clip(centroid + rho * (xr - centroid), -5, 5)
                    else:
                        xc = np.clip(centroid - rho * (centroid - simplex[-1]), -5, 5)
                    fc = evaluate(xc)
                    if fc is None:
                        break
                    if fc < min(fr, fvals[-1]):
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = np.clip(simplex[0] + sigma_shrink * (simplex[i] - simplex[0]), -5, 5)
                        fvals_new = [fvals[0]]
                        for pt in simplex[1:]:
                            if calls >= budget or not time_ok():
                                break
                            fx = evaluate(pt)
                            if fx is None:
                                break
                            fvals_new.append(fx)
                        if len(fvals_new) != len(simplex):
                            break
                        fvals = fvals_new

        # Phase 3: Adaptive (1+1)-ES polish on best point
        if best_x is not None and calls < budget and time_ok():
            x = best_x.copy()
            f = best_f
            sigma = 0.3
            min_sigma = 1e-7
            for _ in range(budget - calls):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = evaluate(y)
                if fy is None:
                    break
                if fy < f:
                    x, f = y, fy
                    sigma = min(sigma * 1.25, 2.0)
                else:
                    sigma *= 0.85
                    if sigma < min_sigma:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
