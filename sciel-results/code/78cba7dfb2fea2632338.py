"""Optimizer for BBOB f7 (Step ellipsoidal).
Hybrid: population-based adaptive-step global search (robust on flat plateaus)
drives exploration; Nelder-Mead simplex refines the best point when budget remains.
A strict wall-clock deadline guard returns the best result found safely early."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based adaptive search with Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        dim = self.dim
        max_calls = self.budget  # 500

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def elapsed_ok():
            return time.time() < deadline and calls_made < max_calls

        # ---------- Nelder-Mead local refinement ----------
        def nelder_mead(func, x0, deadline, rng):
            """Run NM simplex from x0. Updates best_x/best_f/calls_made in place."""
            nonlocal best_x, best_f, calls_made
            n = dim
            if not elapsed_ok():
                return
            alpha = 0.3
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += alpha
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                if not elapsed_ok():
                    return
                fvals[i] = func(simplex[i])
                calls_made += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = simplex[i].copy()

            alpha_r, alpha_e, alpha_c, alpha_sh = 1.0, 2.0, 0.5, 0.5
            max_iter = 120
            tol = 1e-12
            it = 0
            while it < max_iter:
                if not elapsed_ok():
                    return
                it += 1
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]
                if fvals[-1] - fvals[0] < tol:
                    break
                centroid = np.mean(simplex[:-1], axis=0)
                xr = np.clip(centroid + alpha_r * (centroid - simplex[-1]), -5, 5)
                fr = func(xr)
                calls_made += 1
                if fr < fvals[-1]:
                    if fr < fvals[0]:
                        xe = np.clip(centroid + alpha_e * (centroid - simplex[-1]), -5, 5)
                        fe = func(xe)
                        calls_made += 1
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    xc = np.clip(centroid + alpha_c * (simplex[-1] - centroid), -5, 5)
                    fc = func(xc)
                    calls_made += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = np.clip(simplex[0] + alpha_sh * (simplex[i] - simplex[0]), -5, 5)
                        for i in range(1, n + 1):
                            fvals[i] = func(simplex[i])
                            calls_made += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()
                        if not elapsed_ok():
                            return
                mi = np.argmin(fvals)
                if fvals[mi] < best_f:
                    best_f = fvals[mi]
                    best_x = simplex[mi].copy()

        # ---------- Initial exploration: small diverse population ----------
        pop_size = 10
        candidates = []
        for _ in range(pop_size):
            if not elapsed_ok():
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            calls_made += 1
            candidates.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        candidates.sort(key=lambda t: t[0])
        x = candidates[0][1].copy()
        fx = candidates[0][0]

        # ---------- Adaptive-step population main loop ----------
        sigma = 1.5
        min_sigma, max_sigma = 0.01, 5.0
        lam = 10
        stagnation_count = 0
        while elapsed_ok():
            improved = False
            for _ in range(lam):
                if not elapsed_ok():
                    break
                y = np.clip(x + rng.standard_normal(dim) * sigma, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x = y
                    fx = fy
                    improved = True
                    sigma = min(sigma * 1.2, max_sigma)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                else:
                    sigma = max(sigma * 0.9, min_sigma)
            if improved:
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Stagnation: local perturbations around the best point
            if stagnation_count >= 15:
                for _ in range(3):
                    if not elapsed_ok():
                        break
                    y = np.clip(best_x + rng.standard_normal(dim) * 0.5, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        sigma = 1.0
                        stagnation_count = 0
                    elif fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                # Reset counter after a restart attempt
                stagnation_count = 0

            # If step collapsed with no progress, take a big jump to escape
            if sigma < 0.05 and not improved:
                if not elapsed_ok():
                    break
                y = rng.uniform(-5, 5, dim)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x = y
                    fx = fy
                    sigma = 1.5
                elif fy < best_f:
                    best_f = fy
                    best_x = y.copy()

        # ---------- Refine the best point with Nelder-Mead ----------
        if best_x is not None and elapsed_ok():
            nelder_mead(func, best_x.copy(), deadline, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
