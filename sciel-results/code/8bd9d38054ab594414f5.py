"""Rewritten optimizer: adaptive coordinate descent + Nelder-Mead hybrid for Bent Cigar."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Hybrid adaptive coordinate-descent / Nelder-Mead with wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        deadline = t0 + 275.0
        max_calls = budget

        calls = [0]

        def f(x):
            calls[0] += 1
            val = float(func(x))
            if not np.isfinite(val):
                return 1e18
            return val

        def remaining():
            return deadline - time.time()

        def nelder_mead(x, fx, max_calls, max_iter):
            n = dim
            simplex = [np.array(x, dtype=float)]
            for i in range(n):
                p = np.array(x, dtype=float)
                p[i] += 0.5
                simplex.append(p)
            fvals = [fx] + [f(p) for p in simplex[1:]]
            it = 0
            while it < max_iter and calls[0] < max_calls and remaining() > 5.0:
                it += 1
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                worst = simplex[-1]
                best = simplex[0]
                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + 1.0 * (centroid - worst)
                fr = f(xr)
                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    continue
                if fr < fvals[0]:
                    xe = centroid + 2.0 * (centroid - worst)
                    fe = f(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    continue
                if fr < fvals[-1]:
                    xc = centroid + 0.5 * (worst - centroid)
                    fc = f(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                        continue
                else:
                    xc = centroid - 0.5 * (worst - centroid)
                    fc = f(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                        continue
                for i in range(1, n + 1):
                    simplex[i] = best + 0.5 * (simplex[i] - best)
                    fvals[i] = f(simplex[i])
            order = np.argsort(fvals)
            return simplex[order[0]], fvals[order[0]]

        def adaptive_coordinate_descent(x, fx, max_calls, max_rounds):
            """
            Adaptive coordinate descent with step-size shrinking per axis.
            For Bent Cigar, one axis is much steeper; we try larger steps first
            and shrink when no improvement is found.
            """
            n = dim
            steps = [1.0] * n
            for _ in range(max_rounds):
                if calls[0] >= max_calls or remaining() < 5.0:
                    break
                improved_any = False
                for i in range(n):
                    if calls[0] >= max_calls or remaining() < 5.0:
                        break
                    # Try a few step sizes: current, larger, smaller
                    best_xi = x[i]
                    best_fi = fx
                    for scale in [1.0, 2.0, 0.5]:
                        trial_step = steps[i] * scale
                        if trial_step < 1e-8:
                            continue
                        # Try positive and negative steps
                        for sign in [1.0, -1.0]:
                            if calls[0] >= max_calls or remaining() < 5.0:
                                break
                            new_xi = x[i] + sign * trial_step
                            new_xi = np.clip(new_xi, -5.0, 5.0)
                            x[i] = new_xi
                            fi = f(x)
                            if fi < best_fi:
                                best_fi = fi
                                best_xi = new_xi
                        if calls[0] >= max_calls or remaining() < 5.0:
                            break
                    x[i] = best_xi
                    if best_fi < fx:
                        fx = best_fi
                        improved_any = True
                        steps[i] = max(steps[i] * 0.7, 1e-6)
                    else:
                        steps[i] = max(steps[i] * 0.5, 1e-8)
                if not improved_any and max(steps) < 1e-6:
                    break
            return x, fx

        best_x = None
        best_fx = 1e18

        rng = np.random.default_rng(0)
        starts = []
        starts.append(np.zeros(dim))
        starts.append(rng.uniform(-4, 4, dim))
        starts.append(rng.uniform(-4, 4, dim))
        starts.append(rng.uniform(-2, 2, dim))

        for si, s in enumerate(starts):
            if calls[0] >= max_calls or remaining() < 10.0:
                break
            s = np.clip(s, -5, 5)
            fx = f(s)
            # Phase 1: Adaptive coordinate descent (exploits anisotropy)
            x, fx = adaptive_coordinate_descent(s, fx, max_calls, max_rounds=40)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            # Phase 2: Nelder-Mead for final refinement
            if calls[0] < max_calls and remaining() > 10.0:
                x2, fx2 = nelder_mead(x, fx, max_calls, max_iter=150)
                if fx2 < best_fx:
                    best_fx = fx2
                    best_x = x2.copy()
            # Phase 3: Another round of coordinate descent if time permits
            if calls[0] < max_calls and remaining() > 10.0:
                x3, fx3 = adaptive_coordinate_descent(best_x, best_fx, max_calls, max_rounds=20)
                if fx3 < best_fx:
                    best_fx = fx3
                    best_x = x3.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = f(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
