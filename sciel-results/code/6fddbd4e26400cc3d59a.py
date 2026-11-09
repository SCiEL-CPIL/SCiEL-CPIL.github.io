"""Rewritten optimizer: Structure-exploiting grid search + Nelder-Mead for Rastrigin f3."""
import time
import numpy as np


class Optimizer:
    """Structure-exploiting 1D grid search + Nelder-Mead polish for Rastrigin-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0
        dim = self.dim
        budget = self.budget
        self.calls = 0

        def _eval(x):
            if self.calls >= budget:
                return np.inf
            self.calls += 1
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Coarse 1D grid search exploiting Rastrigin's period-1 structure
        # Rastrigin has minima at integer points in the rotated+shifted coordinate frame.
        # A grid with step ~1 along each axis samples near the integer lattice points.
        n_coarse = 11
        coarse_points = np.linspace(-5.0, 5.0, n_coarse)

        # Evaluate at center first
        center = np.zeros(dim)
        update_best(center, _eval(center))

        # Coarse 1D scan along each axis
        for d in range(dim):
            if self.calls >= budget or time.time() > deadline - 30.0:
                break
            for val in coarse_points:
                if abs(val) < 1e-9:
                    continue  # already evaluated center
                if self.calls >= budget:
                    break
                x = best_x.copy()
                x[d] = val
                x = np.clip(x, -5, 5)
                f = _eval(x)
                update_best(x, f)

        # Phase 2: Fine 1D grid search around current best
        # Fine grids with width 0.4 and 0.04 refine near the integer lattice point.
        fine_widths = [0.4, 0.04]
        fine_n = 21

        for width in fine_widths:
            if self.calls >= budget or time.time() > deadline - 20.0:
                break
            for d in range(dim):
                if self.calls >= budget:
                    break
                lo = max(-5.0, best_x[d] - width)
                hi = min(5.0, best_x[d] + width)
                if hi - lo < 1e-6:
                    continue
                fine_pts = np.linspace(lo, hi, fine_n)
                for val in fine_pts:
                    if self.calls >= budget:
                        break
                    x = best_x.copy()
                    x[d] = val
                    x = np.clip(x, -5, 5)
                    f = _eval(x)
                    update_best(x, f)

        # Phase 3: Nelder-Mead polish with multi-start perturbations
        remaining_budget = budget - self.calls
        if remaining_budget > (dim + 1) * 3 and time.time() < deadline - 5.0:
            nm_starts = [best_x.copy()]
            rng = np.random.default_rng(123)
            for _ in range(3):
                perturbed = best_x + rng.normal(0, 0.05, dim)
                perturbed = np.clip(perturbed, -5, 5)
                nm_starts.append(perturbed)

            for start_x in nm_starts:
                if self.calls >= budget or time.time() > deadline - 3.0:
                    break

                simplex = [start_x.copy()]
                for d in range(dim):
                    perturbed = start_x.copy()
                    step = 0.05
                    if perturbed[d] + step > 5.0:
                        step = -step
                    if perturbed[d] + step < -5.0:
                        step = -step
                    perturbed[d] += step
                    simplex.append(perturbed)

                fvals = []
                for s in simplex:
                    if self.calls >= budget:
                        break
                    fvals.append(_eval(s))

                if len(fvals) < dim + 1:
                    continue

                alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
                max_nm_iter = 200

                for _ in range(max_nm_iter):
                    if self.calls >= budget or time.time() > deadline - 2.0:
                        break

                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]

                    if fvals[0] < best_f:
                        best_f = fvals[0]
                        best_x = simplex[0].copy()

                    if abs(fvals[0] - fvals[-1]) < 1e-10:
                        break

                    centroid = np.mean(simplex[:-1], axis=0)

                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)

                    if self.calls >= budget:
                        break
                    fr = _eval(xr)

                    if fvals[0] <= fr < fvals[-2]:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    elif fr < fvals[0]:
                        simplex[-1] = xr
                        fvals[-1] = fr
                        if self.calls >= budget:
                            break
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        fe = _eval(xe)
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    else:
                        if self.calls >= budget:
                            break
                        if fr < fvals[-1]:
                            xc = centroid + rho * (simplex[-1] - centroid)
                        else:
                            xc = centroid - rho * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = _eval(xc)

                        if fc < min(fr, fvals[-1]):
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            for i in range(1, len(simplex)):
                                if self.calls >= budget:
                                    break
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = _eval(simplex[i])

        return best_x, best_f
