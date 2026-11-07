"""Optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with multi-start and adaptive restarts for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(12345)
        n = self.dim
        budget = self.budget

        call_count = [0]

        def eval_func(x):
            if call_count[0] >= budget:
                return np.inf
            call_count[0] += 1
            val = func(x)
            if not np.isfinite(val):
                return np.inf
            return float(val)

        def nelder_mead(func, x0, max_iter=200, initial_step=0.5):
            """Nelder-Mead simplex method with bounds handling."""
            n = len(x0)
            # Initialize simplex
            simplex = [np.copy(x0)]
            f_simplex = [func(x0)]

            for i in range(n):
                xi = np.copy(x0)
                xi[i] += initial_step
                simplex.append(xi)
                f_simplex.append(func(xi))

            # Convert to arrays for efficiency
            simplex = np.array(simplex)
            f_simplex = np.array(f_simplex)

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iter):
                # Sort by function value
                idx = np.argsort(f_simplex)
                simplex = simplex[idx]
                f_simplex = f_simplex[idx]

                # Check convergence
                f_range = f_simplex[-1] - f_simplex[0]
                if f_range < 1e-12 or np.all(np.abs(simplex - simplex[0]) < 1e-10):
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)

                if fr < f_simplex[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                elif fr < f_simplex[-2]:
                    # Simple reflection
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                else:
                    # Contraction
                    if fr < f_simplex[-1]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)

                    if fc < min(fr, f_simplex[-1]):
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])

            # Return best point
            idx_best = np.argmin(f_simplex)
            return simplex[idx_best], f_simplex[idx_best]

        # Multi-start Nelder-Mead
        best_x = None
        best_f = np.inf

        # Number of starts: use budget efficiently
        # Each Nelder-Mead call uses roughly (n+1) + max_iter*(1+extra) evaluations
        # In 5D, each iteration uses ~1-2 evals on average
        # Let's use adaptive number of starts

        # First, do a quick random search to find good starting points
        num_random = min(50, budget // 10)
        random_points = [rng.uniform(-5, 5, n) for _ in range(num_random)]
        random_fvals = [eval_func(p) for p in random_points]

        # Sort random points by fitness
        random_order = np.argsort(random_fvals)
        top_starts = [random_points[i] for i in random_order[:min(10, len(random_order))]]
        top_fvals = [random_fvals[i] for i in random_order[:min(10, len(random_order))]]

        if len(top_fvals) > 0:
            best_f = top_fvals[0]
            best_x = np.copy(top_starts[0])

        # Now run Nelder-Mead from each top starting point
        # Budget remaining for local optimization
        remaining = budget - call_count[0]
        
        # Adaptive: use more budget for better starting points
        # Each NM run uses roughly n+1 + 2*max_iter evaluations
        # With n=5, that's about 6 + 2*max_iter
        # Let's allocate budget proportionally to how good the start is

        # Sort starts by quality
        for i, (x_start, f_start) in enumerate(zip(top_starts, top_fvals)):
            if call_count[0] >= budget - 10:
                break
            if time.time() > deadline:
                break

            # Allocate more iterations to better starting points
            # Use a fraction of remaining budget
            frac = max(0.1, 1.0 - i * 0.08)
            max_iter = int(remaining * frac * 0.5)
            max_iter = max(20, min(max_iter, 150))

            # Adjust initial step based on how far from center
            dist_from_center = np.linalg.norm(x_start)
            initial_step = max(0.1, min(1.0, 2.0 / (1 + dist_from_center)))

            x_opt, f_opt = nelder_mead(eval_func, x_start, max_iter=max_iter, initial_step=initial_step)

            if f_opt < best_f:
                best_f = f_opt
                best_x = np.copy(x_opt)

            remaining = budget - call_count[0]

            # If we've made good progress, do a few more refined runs
            if i < 3 and call_count[0] < budget - 20:
                # Do a finer local search around the best found so far
                if best_x is not None:
                    x_fine = best_x + 0.1 * rng.standard_normal(n)
                    x_fine = np.clip(x_fine, -5, 5)
                    x_opt2, f_opt2 = nelder_mead(eval_func, x_fine, max_iter=50, initial_step=0.05)
                    if f_opt2 < best_f:
                        best_f = f_opt2
                        best_x = np.copy(x_opt2)

        # Final refinement: one more NM run from the global best with small step
        if best_x is not None and call_count[0] < budget - 5:
            x_final = best_x + 0.05 * rng.standard_normal(n)
            x_final = np.clip(x_final, -5, 5)
            x_opt_f, f_opt_f = nelder_mead(eval_func, x_final, max_iter=30, initial_step=0.02)
            if f_opt_f < best_f:
                best_f = f_opt_f
                best_x = np.copy(x_opt_f)

        return best_x, best_f
# EVOLVE-BLOCK-END
