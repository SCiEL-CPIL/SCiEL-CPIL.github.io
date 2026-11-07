"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex with adaptive polish for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_ok():
            return (time.time() - t0) < deadline

        # Phase 1: Multi-start Nelder-Mead
        # We use 12 initial points, top 2 basins get more budget
        n_initial = 12
        candidates = []
        for _ in range(n_initial):
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            candidates.append((fx, x.copy()))

        candidates.sort(key=lambda t: t[0])

        # Allocate remaining budget to top 2 basins
        remaining = budget - n_initial
        # Give 60% to best, 40% to second best (if available)
        if len(candidates) >= 2:
            budget_best = int(remaining * 0.6)
            budget_second = remaining - budget_best
        else:
            budget_best = remaining
            budget_second = 0

        best_x, best_f = candidates[0][1], candidates[0][0]

        # Nelder-Mead for the best candidate
        if budget_best > 0 and time_ok():
            best_x, best_f = self._nelder_mead(
                func, best_x, best_f, budget_best, n, rng, time_ok, deadline
            )

        # Nelder-Mead for the second candidate
        if budget_second > 0 and time_ok() and len(candidates) >= 2:
            x2, f2 = candidates[1][1], candidates[1][0]
            x2, f2 = self._nelder_mead(
                func, x2, f2, budget_second, n, rng, time_ok, deadline
            )
            if f2 < best_f:
                best_x, best_f = x2, f2

        # Phase 2: Adaptive polish with (1+1)-ES style local search
        if time_ok():
            best_x, best_f = self._polish(
                func, best_x, best_f, n, rng, time_ok, deadline
            )

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, budget, n, rng, time_ok, deadline):
        """Nelder-Mead simplex method."""
        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [f0]

        # Create other vertices with small perturbations
        base_step = 0.3
        for i in range(n):
            xi = x0.copy()
            xi[i] += base_step
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            simplex.append(xi)
            fvals.append(fi)
            budget -= 1
            if budget <= 0 or not time_ok():
                break

        # Sort simplex by function value
        idx = np.argsort(fvals)
        simplex = [simplex[i] for i in idx]
        fvals = [fvals[i] for i in idx]

        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        while budget > 0 and time_ok():
            # Sort simplex
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            budget -= 1

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                budget -= 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + 0.5 * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - 0.5 * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                budget -= 1
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        budget -= 1
                        if budget <= 0 or not time_ok():
                            break
                    if budget <= 0 or not time_ok():
                        break

        # Return best point
        idx = np.argmin(fvals)
        return simplex[idx], fvals[idx]

    def _polish(self, func, x0, f0, n, rng, time_ok, deadline):
        """Adaptive local polish using coordinate descent with shrinking steps."""
        x = x0.copy()
        fx = f0
        sigma = 0.2
        no_improve = 0
        max_no_improve = 20

        while time_ok():
            improved = False
            for dim in range(n):
                if not time_ok():
                    break
                # Try positive and negative steps
                for direction in [1, -1]:
                    y = x.copy()
                    y[dim] += direction * sigma
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    if fy < fx - 1e-12:
                        x = y
                        fx = fy
                        improved = True
                        no_improve = 0
                        break  # Move in this direction, try other dims
            if not improved:
                no_improve += 1
                sigma *= 0.5
                if no_improve > max_no_improve or sigma < 1e-8:
                    break
            else:
                sigma *= 1.3  # Grow step size if we're making progress
                sigma = min(sigma, 2.0)

        return x, fx
# EVOLVE-BLOCK-END
