"""Hybrid global-to-local optimizer for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES exploration + Nelder-Mead local exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.time()
        # Safety margin: 300s hard limit, 330s kill -> use 270s budget
        deadline = start + 270.0

        calls = 0
        rng = np.random.default_rng(12345)

        def _clip(x):
            return np.clip(np.asarray(x, dtype=float), -5.0, 5.0)

        def eval_pt(x):
            nonlocal calls
            x = _clip(x)
            calls += 1
            return float(func(x))

        def time_left():
            return deadline - time.time()

        # ---- Phase 1: multi-start (1+1)-ES global exploration ----
        # Spread initial points across the box to hit different basins.
        best_x = None
        best_f = np.inf

        def run_es(x0, f0, max_calls, sigma0):
            nonlocal best_x, best_f, calls
            x = x0.copy()
            fx = f0
            sigma = sigma0
            while calls < max_calls and time_left() > 5.0:
                y = _clip(x + sigma * rng.standard_normal(dim))
                fy = eval_pt(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.85
                    if sigma < 1e-8:
                        sigma = 1e-8
            return x, fx

        # Coarse multi-start: try a handful of spread starting points.
        starts = []
        starts.append(np.zeros(dim))
        for _ in range(9):
            starts.append(rng.uniform(-4.5, 4.5, dim))
        # Deterministic corner-ish spread for robustness
        for c in (-3.0, 0.0, 3.0):
            starts.append(np.full(dim, c))
        starts = np.array(starts[:14])

        # Allocate exploration budget: ~60% of budget to multi-start ES.
        explore_budget = int(budget * 0.55)
        per_start = max(8, explore_budget // len(starts))

        for s in starts:
            if calls >= explore_budget or time_left() < 10.0:
                break
            s = _clip(s)
            # avoid re-evaluating an identical point
            if best_x is not None and np.allclose(s, best_x, atol=1e-6):
                continue
            fs = eval_pt(s)
            if fs < best_f:
                best_f, best_x = fs, s.copy()
            # short local ES from this start
            run_es(s, fs, min(calls + per_start, explore_budget), 1.5)

        if best_x is None:
            best_x = _clip(np.zeros(dim))
            best_f = eval_pt(best_x)

        # ---- Phase 2: Nelder-Mead local exploitation on the best basin ----
        def nelder_mead(x0, f0, max_calls):
            nonlocal best_x, best_f, calls
            # Build initial simplex.
            simplex = [x0.copy()]
            fvals = [f0]
            base_step = 0.5
            for d in range(dim):
                pt = x0.copy()
                if abs(pt[d]) < 1e-12:
                    pt[d] += base_step
                else:
                    pt[d] += base_step * (1.0 if pt[d] > 0 else -1.0)
                pt = _clip(pt)
                fp = eval_pt(pt)
                simplex.append(pt)
                fvals.append(fp)
            simplex = np.array(simplex)
            fvals = np.array(fvals)
            n = dim + 1

            def update_best(x, f):
                nonlocal best_x, best_f
                if f < best_f:
                    best_f, best_x = f, x.copy()

            update_best(x0, f0)

            alpha, gamma, rho, sigma_nm = 1.0, 2.0, -0.5, 0.5
            iters = 0
            max_iters = max_calls * 2
            while calls < max_calls and time_left() > 5.0 and iters < max_iters:
                iters += 1
                # sort
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # best point
                update_best(simplex[0], fvals[0])

                # centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = fvals[-1]

                # reflection
                xr = _clip(centroid + alpha * (centroid - worst))
                fr = eval_pt(xr)
                if fr < fvals[0]:
                    # expansion
                    xe = _clip(centroid + gamma * (centroid - worst))
                    fe = eval_pt(xe)
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                elif fr < fvals[-2]:
                    simplex[-1], fvals[-1] = xr, fr
                else:
                    # contraction
                    if fr < worst_f:
                        xc = _clip(centroid + rho * (worst - centroid))
                    else:
                        xc = _clip(centroid - rho * (worst - centroid))
                    fc = eval_pt(xc)
                    if fc < min(fr, worst_f):
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        # shrink toward best
                        for i in range(1, n):
                            simplex[i] = _clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            fvals[i] = eval_pt(simplex[i])
                        fvals[0] = fvals[0]  # unchanged
                        # re-evaluate best? keep as is
                # guard against degenerate simplex
                spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                if np.max(spread) < 1e-10:
                    # kick
                    simplex = x0[None, :].repeat(n, axis=0)
                    for i in range(1, n):
                        simplex[i] = _clip(simplex[0] + 1e-3 * rng.standard_normal(dim))
                    fvals = np.array([eval_pt(simplex[i]) for i in range(n)])
                    update_best(simplex[0], fvals[0])
            return simplex[0], fvals[0]

        # Run NM from the best found; also from a couple of good alternates if time.
        # We only track the single best here; re-run NM from best_x.
        nelder_mead(best_x.copy(), best_f, budget)

        # If time and calls remain, do a final fine (1+1)-ES polish from the best.
        if time_left() > 5.0 and calls < budget:
            run_es(best_x.copy(), best_f, budget, 0.3)

        return best_x, best_f
# EVOLVE-BLOCK-END
