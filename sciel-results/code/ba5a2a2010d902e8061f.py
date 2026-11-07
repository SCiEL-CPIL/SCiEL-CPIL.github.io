"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, multi-start, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or not time_ok():
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                f = best_f
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        n = dim

        # Phase 1: Global random sampling to find good basins
        n_initial = min(15, budget // 20)
        candidates = []
        for i in range(n_initial):
            if calls >= budget or not time_ok():
                break
            x = rng.uniform(-5, 5, n)
            fx = safe_call(x)
            if fx is not None:
                candidates.append((fx, x.copy()))

        if not candidates:
            x = rng.uniform(-5, 5, n)
            safe_call(x)
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])

        # Phase 2: Nelder-Mead local refinement on top candidates
        top_k = min(3, len(candidates))
        remaining = budget - calls
        per_candidate = max(50, remaining // max(1, top_k + 1))

        for rank in range(top_k):
            if calls >= budget or not time_ok():
                break

            f0, x0 = candidates[rank]
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            # Build simplex with adaptive initial steps
            simplex = [x0.copy()]
            for j in range(n):
                point = x0.copy()
                # Choose delta based on distance to boundary
                dist_to_pos = 5.0 - x0[j]
                dist_to_neg = x0[j] + 5.0
                if dist_to_pos > 0.5:
                    delta = min(0.5, dist_to_pos)
                elif dist_to_neg > 0.5:
                    delta = -min(0.5, dist_to_neg)
                else:
                    delta = 0.5 if dist_to_pos > dist_to_neg else -0.5
                point[j] += delta
                point = np.clip(point, -5, 5)
                simplex.append(point)

            fvals = [f0]
            for pt in simplex[1:]:
                if calls >= budget or not time_ok():
                    break
                fx = safe_call(pt)
                if fx is None:
                    break
                fvals.append(fx)

            if len(fvals) != len(simplex):
                continue

            max_nm_iter = per_candidate - (len(fvals) - 1)
            for _ in range(max(0, max_nm_iter)):
                if calls >= budget or not time_ok():
                    break

                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if abs(fvals[-1] - fvals[0]) < 1e-12:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                if calls >= budget or not time_ok():
                    break
                fr = safe_call(xr)
                if fr is None:
                    break

                if fr < fvals[0]:
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    if calls >= budget or not time_ok():
                        break
                    fe = safe_call(xe)
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
                    if calls >= budget or not time_ok():
                        break
                    fc = safe_call(xc)
                    if fc is None:
                        break
                    if fc < min(fr, fvals[-1]):
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = np.clip(simplex[0] + sigma_shrink * (simplex[i] - simplex[0]), -5, 5)
                            if calls >= budget or not time_ok():
                                break
                        if calls >= budget or not time_ok():
                            break
                        fvals_new = []
                        for pt in simplex:
                            if pt is simplex[0]:
                                fvals_new.append(fvals[0])
                            else:
                                fx = safe_call(pt)
                                if fx is None:
                                    break
                                fvals_new.append(fx)
                        if len(fvals_new) != len(simplex):
                            break
                        fvals = fvals_new

        # Phase 3: Final (1+1)-ES polish on the best point found
        if calls < budget and time_ok():
            x = best_x.copy()
            f = best_f
            sigma = 0.1
            min_sigma = 1e-8
            for _ in range(budget - calls):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                if fy < f:
                    x, f = y, fy
                    sigma = min(sigma * 1.2, 1.0)
                else:
                    sigma *= 0.9
                    if sigma < min_sigma:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
