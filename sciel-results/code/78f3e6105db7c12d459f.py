"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with Rosenbrock-aware initialization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, remaining_budget, deadline):
        """Run Nelder-Mead from x0, using at most remaining_budget calls."""
        dim = self.dim
        n = dim + 1
        # Build initial simplex
        simplex = np.zeros((n, dim))
        simplex[0] = x0
        for i in range(1, n):
            simplex[i] = x0.copy()
            simplex[i, i - 1] = x0[i - 1] + 0.5

        # Evaluate simplex
        fvals = np.zeros(n)
        calls_used = 0
        for i in range(n):
            if time.time() > deadline:
                break
            fvals[i] = func(simplex[i])
            calls_used += 1
            if calls_used >= remaining_budget:
                break

        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx].copy()
        best_f = fvals[best_idx]

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        while calls_used < remaining_budget and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            best_idx = 0
            worst_idx = n - 1
            second_worst_idx = n - 2

            # Check convergence
            if np.abs(fvals[worst_idx] - fvals[best_idx]) < 1e-12:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:n-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[worst_idx])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_used += 1

            if fr < fvals[0]:
                # Try expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls_used += 1
                if fe < fr:
                    simplex[worst_idx] = xe
                    fvals[worst_idx] = fe
                else:
                    simplex[worst_idx] = xr
                    fvals[worst_idx] = fr
            elif fr < fvals[second_worst_idx]:
                simplex[worst_idx] = xr
                fvals[worst_idx] = fr
            else:
                # Contraction
                if fr < fvals[worst_idx]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (xr - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls_used += 1
                if fc <= fr:
                    simplex[worst_idx] = xc
                    fvals[worst_idx] = fc
                else:
                    # Shrink
                    for i in range(1, n):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls_used += 1
                        if calls_used >= remaining_budget:
                            break

            # Update best
            current_best_idx = np.argmin(fvals)
            if fvals[current_best_idx] < best_f:
                best_f = fvals[current_best_idx]
                best_x = simplex[current_best_idx].copy()

        return best_x, best_f, calls_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget

        # Rosenbrock-aware initialization: try points near the valley
        # The Rosenbrock valley is near x_i = 1 for i > 0, x_0 free
        # But we have hidden shift/rotation, so use multiple strategies

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Strategy 1: Try a few diverse starting points
        start_points = []
        # Center
        start_points.append(np.zeros(dim))
        # Rosenbrock-like: [1, 1, 1, 1, 1]
        start_points.append(np.ones(dim))
        # Random points in [-3, 3]
        for _ in range(8):
            start_points.append(rng.uniform(-3, 3, dim))

        # Allocate budget: give more to promising starts
        # First evaluate all starts, then refine the best ones
        for sp in start_points:
            if time.time() > deadline or calls_made >= total_budget - 50:
                break
            sp = np.clip(sp, -5, 5)
            f = func(sp)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = sp.copy()

        # Now run Nelder-Mead from the best start and a few others
        # Sort start points by their function values (we didn't store them, so re-evaluate top candidates)
        # Actually, let's just run NM from best_x and a couple of random perturbations
        nm_starts = [best_x.copy()]
        # Add a few perturbed versions
        for _ in range(3):
            pert = best_x + rng.uniform(-1, 1, dim)
            nm_starts.append(np.clip(pert, -5, 5))

        for ns in nm_starts:
            if time.time() > deadline or calls_made >= total_budget - 10:
                break
            remaining = total_budget - calls_made
            if remaining < 10:
                break
            nx, nf, used = self._nelder_mead(func, ns, remaining, deadline)
            calls_made += used
            if nf < best_f:
                best_f = nf
                best_x = nx.copy()

        # If we still have budget, do a final polish with a small step size
        if time.time() < deadline and calls_made < total_budget - 5 and best_x is not None:
            # Small random search around best
            sigma = 0.1
            for _ in range(min(20, total_budget - calls_made)):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 0.7
                else:
                    sigma *= 0.9

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
