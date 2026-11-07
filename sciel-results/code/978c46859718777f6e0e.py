"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder–Mead with adaptive step size, preceded by a cheap coarse scan."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, fx0, budget_left, deadline, state):
        n = self.dim
        calls = state["calls"]
        best_x = state["best_x"]
        best_f = state["best_f"]

        # Build initial simplex
        simplex = [x0.copy()]
        fvals = [fx0]
        for i in range(n):
            d = x0.copy()
            d[i] += 0.5
            d = np.clip(d, -5.0, 5.0)
            simplex.append(d)
            fvals.append(func(d))
            calls += 1
            if fvals[-1] < best_f:
                best_f = fvals[-1]
                best_x = simplex[-1].copy()
        state["calls"] = calls

        # Track a step size that adapts to progress
        step = 0.5
        stall = 0

        while calls < budget_left:
            if time.time() > deadline:
                break

            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + step * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = func(xr)
            calls += 1
            if fr < best_f:
                best_f = fr
                best_x = xr.copy()
            state["calls"] = calls

            if fr < fvals[0]:
                # Reflection good: expand
                xe = centroid + 2.0 * (xr - centroid)
                xe = np.clip(xe, -5.0, 5.0)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
                if fe < best_f:
                    best_f = fe
                    best_x = simplex[-1].copy()
                state["calls"] = calls
                step = min(step * 1.5, 4.0)
                stall = 0
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
                stall = 0
            else:
                # Contraction
                if fr < fvals[-1]:
                    xc = centroid + 0.5 * (xr - centroid)
                    fc = func(xc)
                    calls += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    state["calls"] = calls
                    if fc < fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                        stall = 0
                    else:
                        step *= 0.5
                        stall += 1
                else:
                    # Shrink toward best point
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if fvals[i] < best_f:
                            best_f = fvals[i]
                            best_x = simplex[i].copy()
                    state["calls"] = calls
                    step *= 0.5
                    stall += 1

            if stall >= 3:
                # Reset simplex around best point with fresh step
                step = 1.0
                stall = 0
                simplex = [best_x.copy()]
                fvals = [best_f]
                for i in range(n):
                    d = best_x.copy()
                    d[i] += 0.5
                    d = np.clip(d, -5.0, 5.0)
                    simplex.append(d)
                    fvals.append(func(d))
                    calls += 1
                    if fvals[-1] < best_f:
                        best_f = fvals[-1]
                        best_x = simplex[-1].copy()
                state["calls"] = calls

        return best_x, best_f

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0  # safety margin below 300s/330s
        n = self.dim
        state = {"calls": 0, "best_x": None, "best_f": np.inf}

        # --- Phase 1: cheap coarse grid to get early progress ---
        # A 3-point grid along each axis (0, 2, 4) plus center gives 25 points.
        grid_vals = [-4.0, 0.0, 4.0]
        xs = []
        for a in grid_vals:
            for b in grid_vals:
                for c in grid_vals:
                    for d in grid_vals:
                        for e in grid_vals:
                            xs.append([a, b, c, d, e])
        # Add a few random starts
        rng = np.random.default_rng(0)
        for _ in range(15):
            xs.append(list(rng.uniform(-4.0, 4.0, n)))

        for x in xs:
            if time.time() > deadline:
                break
            if state["calls"] >= self.budget:
                break
            x = np.array(x, dtype=float)
            fx = func(x)
            state["calls"] += 1
            if fx < state["best_f"]:
                state["best_f"] = fx
                state["best_x"] = x.copy()

        if state["best_x"] is None:
            # Fallback
            x0 = np.zeros(n)
            fx0 = func(x0)
            state["calls"] += 1
            state["best_f"] = fx0
            state["best_x"] = x0.copy()

        # --- Phase 2: Nelder–Mead from the best point ---
        self._nelder_mead(func, state["best_x"], state["best_f"],
                          self.budget, deadline, state)

        return state["best_x"], state["best_f"]
# EVOLVE-BLOCK-END
