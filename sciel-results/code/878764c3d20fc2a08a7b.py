"""Improved BBOB f22 optimizer: Multi-Start (1+1)-ES + Nelder-Mead polish."""
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with Nelder-Mead polish for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 285.0  # safety margin before 300s kill

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        rng = np.random.default_rng(12345)

        def nelder_mead(x0, f0, max_iter, tol=1e-10):
            """Nelder-Mead simplex method."""
            nonlocal best_x, best_f, calls_made

            n = self.dim
            # Build initial simplex
            simplex = [x0.copy()]
            fvals = [f0]

            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.1
                xi = np.clip(xi, -5, 5)
                fi = safe_func(xi)
                simplex.append(xi)
                fvals.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for iteration in range(max_iter):
                if not check_time() or (self.budget - calls_made) <= 0:
                    break

                # Sort by function value
                idx = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in idx]
                fvals = [fvals[i] for i in idx]

                # Check convergence
                if abs(fvals[-1] - fvals[0]) < tol:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = safe_func(xr)
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    fe = safe_func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                        if fe < best_f:
                            best_f = fe
                            best_x = xe.copy()
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        xc = centroid - rho * (xc if False else (simplex[-1] - centroid))
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = safe_func(xc)
                    if fc < max(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                        if fc < best_f:
                            best_f = fc
                            best_x = xc.copy()
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = safe_func(simplex[i])
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()

            return best_x, best_f

        def one_plus_one_es(x0, f0, max_iters, sigma_init=0.5):
            """(1+1)-ES with adaptive step size."""
            nonlocal best_x, best_f, calls_made

            x = x0.copy()
            f = f0
            sigma = sigma_init
            success = 0.0
            target = 1.0 / (self.dim + 2)
            rate = 0.5

            for it in range(max_iters):
                if not check_time() or (self.budget - calls_made) <= 0:
                    break

                # Generate offspring
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = safe_func(y)

                if fy < f:
                    x = y
                    f = fy
                    success = 1.0
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    success = 0.0

                # Adaptive sigma
                sigma *= np.exp(rate * (success - target))
                sigma = np.clip(sigma, 1e-8, 5.0)

                if sigma < 1e-7:
                    break

            return x, f

        # Phase 1: Random sampling to find good basins
        n_random = min(40, self.budget // 12)
        random_points = []
        for i in range(n_random):
            if not check_time() or calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_func(x)
            random_points.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        random_points.sort(key=lambda t: t[0])

        # Phase 2: (1+1)-ES from top random points
        n_es = min(5, len(random_points))
        for idx in range(n_es):
            if not check_time() or (self.budget - calls_made) <= 0:
                break
            x_start = random_points[idx][1].copy()
            f_start = random_points[idx][0]
            # More iterations for better starting points
            max_iters = max(20, (self.budget - calls_made) // max(1, n_es - idx))
            one_plus_one_es(x_start, f_start, max_iters, sigma_init=0.3)

        # Phase 3: Nelder-Mead polish from best point
        if best_x is not None and check_time() and (self.budget - calls_made) > 10:
            nm_iter = min(100, (self.budget - calls_made) // 2)
            nelder_mead(best_x.copy(), best_f, nm_iter)

        # Phase 4: Final (1+1)-ES polish
        if best_x is not None and check_time() and (self.budget - calls_made) > 0:
            remaining = self.budget - calls_made
            one_plus_one_es(best_x.copy(), best_f, min(remaining, 80), sigma_init=0.05)

        return best_x, best_f
