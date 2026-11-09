"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset.

Uses multi-start Nelder-Mead initialized from Sobol space-filling points,
then focused convergence on the best basin.
"""
import time
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """(1+1)-ES with restarts and Nelder-Mead polish for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 310.0

        dim = self.dim
        budget = self.budget

        def still_ok():
            return time.time() < deadline

        def nelder_mead(func, x0, f0, calls_remaining, max_iter=40, step=0.3):
            """Nelder-Mead local optimization with early exit."""
            if calls_remaining < 12:
                return x0, f0, 0

            # Build initial simplex
            simplex = [(x0.copy(), f0)]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += step
                xi = np.clip(xi, -5.0, 5.0)
                fi = float(func(xi))
                calls_remaining -= 1
                simplex.append((xi, fi))

            simplex.sort(key=lambda t: t[1])

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            calls_used = dim + 1
            for _ in range(max_iter):
                if calls_remaining - calls_used < 5:
                    break

                centroid = np.mean([s[0] for s in simplex[:dim]], axis=0)
                worst = simplex[dim]

                xr = centroid + alpha * (centroid - worst[0])
                xr = np.clip(xr, -5.0, 5.0)
                fr = float(func(xr))
                calls_used += 1

                if fr < simplex[0][1]:
                    xe = centroid + gamma * (centroid - worst[0])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = float(func(xe))
                    calls_used += 1
                    simplex[dim] = (xe, fe) if fe < fr else (xr, fr)
                elif fr < simplex[dim - 1][1]:
                    simplex[dim] = (xr, fr)
                else:
                    if fr < simplex[dim - 1][1]:
                        xc = centroid + rho * (worst[0] - centroid)
                    else:
                        xc = centroid - rho * (centroid - worst[0])
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = float(func(xc))
                    calls_used += 1

                    if fc < simplex[dim - 1][1]:
                        simplex[dim] = (xc, fc)
                    else:
                        for i in range(1, dim + 1):
                            simplex[i][0] = simplex[0][0] + sigma_nm * (simplex[i][0] - simplex[0][0])
                            simplex[i][0] = np.clip(simplex[i][0], -5.0, 5.0)
                            simplex[i][1] = float(func(simplex[i][0]))
                            calls_used += 1

                simplex.sort(key=lambda t: t[1])
                if abs(simplex[0][1] - simplex[dim][1]) < 1e-13:
                    break

            return simplex[0][0], simplex[0][1], calls_used

        # Generate Sobol points for structured space-filling initialization
        n_sobol = 32
        sobol_sampler = qmc.Sobol(d=dim, scramble=True, seed=123)
        # Sobol generates in [0,1]^d, map to [-5,5]^d
        sobol_unit = sobol_sampler.random(n_sobol)
        sobol_points = sobol_unit * 10.0 - 5.0

        # Add a few hand-picked strategic points
        strategic = [
            np.zeros(dim),
            np.array([4.0] * dim),
            np.array([-4.0] * dim),
            np.array([3.0, -3.0, 3.0, -3.0, 3.0]),
            np.array([-3.0, 3.0, -3.0, 3.0, -3.0]),
        ]
        all_starts = [p.copy() for p in sobol_points] + [p.copy() for p in strategic]

        best_x_overall = np.zeros(dim)
        best_f_overall = np.inf
        calls_made = 0

        # Phase 1: Evaluate all starting points, track best
        start_results = []
        for x in all_starts:
            if calls_made >= budget or not still_ok():
                break
            f = float(func(x))
            calls_made += 1
            start_results.append((x.copy(), f))
            if f < best_f_overall:
                best_f_overall = f
                best_x_overall = x.copy()

        # Phase 2: Multi-start Nelder-Mead — run NM from top-K distinct basins
        # Sort starts by function value, pick top-K with minimum pairwise distance
        start_results.sort(key=lambda t: t[1])
        K = min(6, len(start_results))
        min_dist = 1.5  # minimum distance between selected starts

        selected = [start_results[0]]
        for candidate in start_results[1:]:
            if len(selected) >= K:
                break
            if calls_made >= budget or not still_ok():
                break
            # Check distance to all selected
            too_close = False
            for sel in selected:
                if np.linalg.norm(candidate[0] - sel[0]) < min_dist:
                    too_close = True
                    break
            if not too_close:
                selected.append(candidate)

        # Run Nelder-Mead from each selected start
        for idx, (x_start, f_start) in enumerate(selected):
            if calls_made >= budget or not still_ok():
                break
            # Allocate remaining budget proportionally, give more to earlier (better) starts
            remaining = budget - calls_made
            # Each NM needs at least (dim+1) + some iterations; cap per-start budget
            per_start_budget = max(15, min(remaining // (K - idx), 80))
            if per_start_budget < 12:
                break

            x_nm, f_nm, calls_used = nelder_mead(
                func, x_start, f_start, per_start_budget, max_iter=35, step=0.3
            )
            calls_made += calls_used
            if f_nm < best_f_overall:
                best_f_overall = f_nm
                best_x_overall = x_nm.copy()

        # Phase 3: Final polish on the best point with larger iteration budget
        if still_ok() and calls_made < budget:
            remaining = budget - calls_made
            if remaining >= 15:
                x_final, f_final, calls_used = nelder_mead(
                    func, best_x_overall, best_f_overall, remaining, max_iter=60, step=0.15
                )
                calls_made += calls_used
                if f_final < best_f_overall:
                    best_f_overall = f_final
                    best_x_overall = x_final.copy()

        # Phase 4: If budget remains, do a fine-grained local search around best
        if still_ok() and calls_made < budget:
            # Coordinate descent style: try small steps in each direction
            step_sizes = [0.1, 0.05, 0.02]
            for ss in step_sizes:
                if calls_made >= budget or not still_ok():
                    break
                for d in range(dim):
                    if calls_made >= budget or not still_ok():
                        break
                    for sign in [1.0, -1.0]:
                        if calls_made >= budget or not still_ok():
                            break
                        x_try = best_x_overall.copy()
                        x_try[d] += sign * ss
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = float(func(x_try))
                        calls_made += 1
                        if f_try < best_f_overall:
                            best_f_overall = f_try
                            best_x_overall = x_try.copy()

        return best_x_overall, float(best_f_overall)
