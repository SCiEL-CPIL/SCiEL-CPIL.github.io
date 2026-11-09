"""Optimiser for the BBOB-style f8 (Rosenbrock) task.

Fresh design: adaptive Nelder-Mead simplex (derivative-free, well suited to the
curved Rosenbrock valley) plus a CMA-ES-like adaptive step, with random
restarts on stagnation and an explicit wall-clock deadline guard.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Adaptive Nelder-Mead + adaptive step, with restarts and a time guard."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)
        self.deadline = time.time() + 270.0   # safety margin well under 300 s

    # ------------------------------------------------------------------ #
    def _nm_step(self, simplex, fx, sigma, rng):
        """One Nelder-Mead iteration. Returns (simplex, fx, improved, sigma)."""
        n = self.dim
        order = np.argsort(fx)
        simplex = simplex[order]
        fx = fx[order]

        # reflect
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = self._eval(xr)
        if fr < fx[0]:
            # expand
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = self._eval(xe)
            if fe < fr:
                simplex[-1] = xe
                fx[-1] = fe
            else:
                simplex[-1] = xr
                fx[-1] = fr
            improved = True
        elif fr < fx[-2]:
            simplex[-1] = xr
            fx[-1] = fr
            improved = True
        else:
            # contract
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = self._eval(xc)
            if fc < fr:
                simplex[-1] = xc
                fx[-1] = fc
                improved = True
            else:
                # shrink
                pivot = simplex[0]
                simplex[1:] = pivot + 0.5 * (simplex[1:] - pivot)
                for i in range(1, n + 1):
                    fx[i] = self._eval(simplex[i])
                improved = False
                sigma *= 0.5
        return simplex, fx, improved, sigma

    # ------------------------------------------------------------------ #
    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        calls = 0

        def _eval(x):
            nonlocal calls
            calls += 1
            return float(func(np.clip(x, -5.0, 5.0)))

        # ---- initial point: a few random starts, keep the best ----------
        x = np.zeros(dim)
        best_fx = np.inf
        for _ in range(8):
            c = rng.uniform(-5.0, 5.0, dim)
            fc = _eval(c)
            if fc < best_fx:
                best_fx, x = fc, c.copy()
        budget_left = budget - calls
        if budget_left <= 0:
            return x, best_fx

        # ---- build initial simplex around the best point -----------------
        simplex = np.empty((dim + 1, dim))
        fx = np.empty(dim + 1)
        simplex[0] = x
        fx[0] = best_fx
        for i in range(1, dim + 1):
            v = x.copy()
            v[i - 1] += 1.0
            simplex[i] = v
            fx[i] = _eval(v)
        budget_left = budget - calls
        if budget_left <= 0:
            return x, best_fx

        sigma = 1.0
        no_improve = 0
        best_seen = best_fx

        # ---- main loop: NM iterations with adaptive step / restarts -----
        while budget_left > 0:
            if time.time() > self.deadline:
                break

            simplex, fx, improved, sigma = self._nm_step(simplex, fx, sigma, rng)
            budget_left = budget - calls
            if budget_left <= 0:
                break

            idx = int(np.argmin(fx))
            if fx[idx] < best_seen:
                best_seen = fx[idx]
                no_improve = 0
            else:
                no_improve += 1

            # keep the best point
            x = simplex[idx].copy()
            best_fx = fx[idx]

            # restart if long stagnation
            if no_improve > 40 and budget_left > 30:
                no_improve = 0
                sigma = max(0.3, 1.0)
                for _ in range(4):
                    c = rng.uniform(-5.0, 5.0, dim)
                    fc = _eval(c)
                    if fc < best_seen:
                        best_seen, x = fc, c.copy()
                        simplex[0] = x
                        fx[0] = fc
                # rebuild simplex around best
                if best_seen < fx[0]:
                    simplex[0] = x
                    fx[0] = best_seen
                for i in range(1, dim + 1):
                    v = x.copy()
                    v[i - 1] += 1.0
                    simplex[i] = v
                    fx[i] = _eval(v)
                budget_left = budget - calls

        # final: return best point found
        return x, best_fx
# EVOLVE-BLOCK-END
