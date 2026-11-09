"""Combined optimizer: LHS initialization + (1+1)-ES with Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with LHS initialization, adaptive sigma, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        bounds = np.array([-5.0] * dim)
        upper = np.array([5.0] * dim)

        # Phase 1: Latin Hypercube-style sampling for fast early progress
        lhs_budget = min(50, budget // 10)
        x_best = None
        fx_best = np.inf

        # Generate LHS points
        lhs_points = np.zeros((lhs_budget, dim))
        for i in range(dim):
            perm = rng.permutation(lhs_budget)
            lhs_points[:, i] = (perm + 0.5) / lhs_budget * 10.0 - 5.0

        # Evaluate LHS points
        for i in range(lhs_budget):
            x = lhs_points[i]
            fx = func(x)
            if fx < fx_best:
                fx_best = fx
                x_best = x.copy()

        # Check deadline
        if time.time() > deadline:
            return x_best, fx_best

        # Phase 2: (1+1)-ES with adaptive sigma and periodic restarts
        remaining_budget = budget - lhs_budget
        sigma = 1.0
        x = x_best.copy()
        fx = fx_best

        # Track best for restart decisions
        fx_global_best = fx
        x_global_best = x.copy()
        no_improve_count = 0
        restart_threshold = 80
        max_sigma = 5.0
        min_sigma = 0.01

        # 1/5 success rule parameters
        sigma_up = 1.34
        sigma_down = 0.8
        success_window = 5
        successes = 0
        window = 0

        calls_made = lhs_budget

        for step in range(remaining_budget):
            # Deadline check every 10 steps for efficiency
            if step % 10 == 0 and time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, bounds, upper)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                no_improve_count = 0
                successes += 1
                window += 1

                # Adaptive sigma using 1/5 success rule
                if window >= success_window:
                    if successes / window > 0.3:
                        sigma = min(sigma * sigma_up, max_sigma)
                    elif successes / window < 0.1:
                        sigma = max(sigma * sigma_down, min_sigma)
                    successes = 0
                    window = 0

                # Update global best
                if fx < fx_global_best:
                    fx_global_best = fx
                    x_global_best = x.copy()
            else:
                no_improve_count += 1
                window += 1
                if window >= success_window:
                    if successes / window < 0.1:
                        sigma = max(sigma * sigma_down, min_sigma)
                    successes = 0
                    window = 0

            # Periodic restart: if stuck for too long, restart from global best
            if no_improve_count >= restart_threshold:
                no_improve_count = 0
                x = x_global_best.copy()
                fx = fx_global_best
                sigma = max_sigma * 0.5

        # Phase 3: Nelder-Mead polish around the best point
        if x_global_best is not None and time.time() < deadline and calls_made < budget:
            nm_budget = min(budget - calls_made, 50)
            if nm_budget > dim + 1:
                x_nm, f_nm, calls_nm = self._nm_polish(func, x_global_best, dim, nm_budget, rng)
                calls_made += calls_nm
                if f_nm < fx_global_best:
                    fx_global_best = f_nm
                    x_global_best = x_nm

        return x_global_best, fx_global_best

    def _nm_polish(self, func, x0, dim, budget, rng, lo=-5.0, hi=5.0):
        """Nelder-Mead with box projection; returns (best_x, best_f, calls_used)."""
        x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)
        f0 = func(x0)
        calls = 1
        # Initial simplex
        xk = [x0.copy()]
        for j in range(dim):
            xi = x0.copy()
            step = 0.25 if x0[j] < 0 else -0.25
            xi[j] += step
            xi = np.clip(xi, lo, hi)
            xk.append(xi)
        fx = [f0] + [func(xi) for xi in xk[1:]]
        calls += dim

        for _ in range(budget - calls):
            if calls >= budget:
                break
            order = np.argsort(fx)
            xk = [xk[i] for i in order]
            fx = [fx[i] for i in order]
            if calls >= budget:
                break

            # centroid of all but worst
            centroid = np.mean(xk[:-1], axis=0)
            worst = xk[-1]
            # reflection
            xr = np.clip(centroid + 1.0 * (centroid - worst), lo, hi)
            fr = func(xr)
            calls += 1
            if fr < fx[-2]:
                # expansion
                xe = np.clip(centroid + 2.0 * (xr - centroid), lo, hi)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    xk[-1], fx[-1] = xe, fe
                else:
                    xk[-1], fx[-1] = xr, fr
            elif fr <= fx[0]:
                xk[-1], fx[-1] = xr, fr
            else:
                # contraction
                xc = np.clip(centroid + 0.5 * (worst - centroid), lo, hi)
                fc = func(xc)
                calls += 1
                if fc < fx[-1]:
                    xk[-1], fx[-1] = xc, fc
                else:
                    # shrink toward best
                    for i in range(1, dim + 1):
                        xk[i] = np.clip(xk[0] + 0.5 * (xk[i] - xk[0]), lo, hi)
                        fx[i] = func(xk[i])
                        calls += 1
                        if calls >= budget:
                            break
            if calls >= budget:
                break

        order = np.argsort(fx)
        best_i = order[0]
        return xk[best_i], fx[best_i], calls
# EVOLVE-BLOCK-END
