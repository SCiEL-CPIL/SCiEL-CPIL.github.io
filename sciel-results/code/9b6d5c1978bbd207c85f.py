"""Hybrid global-local optimizer for f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Global population search + Nelder-Mead local refinement with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = 325.0  # safety margin below 330s kill

    def __call__(self, func):
        start = time.time()
        rng = np.random.default_rng(0)
        dim = self.dim
        lo, hi = -5.0, 5.0

        # ---------- Global phase: population-based (1+lambda)-ES ----------
        # Use a moderate population for global coverage of the shifted/rotated landscape.
        pop_size = 12
        sigma = 2.0
        x = rng.uniform(lo, hi, dim)
        fx = float(func(x))
        best_x, best_f = x.copy(), fx
        calls = 1

        # Track progress for AOCC (early progress counts)
        best_history = [best_f]

        def elapsed():
            return time.time() - start

        # Global search loop
        while calls < self.budget - 50 and elapsed() < self.deadline - 30.0:
            # Generate offspring
            for j in range(pop_size):
                if calls >= self.budget:
                    break
                if elapsed() > self.deadline - 30.0:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), lo, hi)
                fy = float(func(y))
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.9
                if fx < best_f:
                    best_x, best_f = x.copy(), fx
            # Adaptive sigma: shrink if no improvement in this generation
            if elapsed() > self.deadline - 30.0:
                break
            # Occasionally re-center search around best point
            if rng.random() < 0.15:
                x = best_x + 0.5 * sigma * rng.standard_normal(dim)
                x = np.clip(x, lo, hi)
                fx = float(func(x))
                calls += 1
                if fx < best_f:
                    best_x, best_f = x.copy(), fx

        # ---------- Local phase: Nelder-Mead simplex refinement ----------
        # Run Nelder-Mead from the best point found, and from a few other good candidates.
        # This exploits the smooth local structure of Griewank-Rosenbrock.
        nm_start_points = [best_x.copy()]

        # Add a few diverse starting points for local search
        for _ in range(3):
            sp = best_x + 0.3 * rng.standard_normal(dim)
            sp = np.clip(sp, lo, hi)
            nm_start_points.append(sp)

        for sp in nm_start_points:
            if elapsed() > self.deadline - 15.0:
                break
            if calls >= self.budget:
                break
            nm_best_x, nm_best_f = self._nelder_mead(
                func, sp, dim, lo, hi, budget_left=self.budget - calls,
                deadline=self.deadline, start_time=start
            )
            # Update call count (approximate; Nelder-Mead tracks internally)
            calls = self.budget  # mark as used; we guard by deadline inside
            if nm_best_f < best_f:
                best_x, best_f = nm_best_x, nm_best_f

        return best_x, best_f

    def _nelder_mead(self, func, x0, dim, lo, hi, budget_left, deadline, start_time):
        """Nelder-Mead simplex method with box constraints and deadline guard."""
        if budget_left <= 0:
            return x0.copy(), float(func(x0))

        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [float(func(x0))]
        calls = 1
        best_x, best_f = x0.copy(), fvals[0]

        # Expand simplex in each dimension
        for i in range(dim):
            if calls >= budget_left:
                break
            if time.time() - start_time > deadline - 10.0:
                break
            xi = x0.copy()
            step = 0.1 if x0[i] >= 0 else -0.1
            xi[i] += step
            xi = np.clip(xi, lo, hi)
            fi = float(func(xi))
            calls += 1
            simplex.append(xi.copy())
            fvals.append(fi)
            if fi < best_f:
                best_x, best_f = xi.copy(), fi

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        max_iter = max(1, budget_left - calls)

        for _ in range(max_iter):
            if calls >= budget_left:
                break
            if time.time() - start_time > deadline - 10.0:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Update best
            if fvals[0] < best_f:
                best_x, best_f = simplex[0].copy(), fvals[0]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, lo, hi)
            fr = float(func(xr))
            calls += 1

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, lo, hi)
                fe = float(func(xe))
                calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Accept reflection
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                xc = np.clip(xc, lo, hi)
                fc = float(func(xc))
                calls += 1
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lo, hi)
                        fvals[i] = float(func(simplex[i]))
                        calls += 1
                        if fvals[i] < best_f:
                            best_x, best_f = simplex[i].copy(), fvals[i]

        return best_x, best_f
# EVOLVE-BLOCK-END
