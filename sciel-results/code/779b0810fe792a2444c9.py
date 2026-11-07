"""Stochastic multi-start Nelder-Mead with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead on [-5,5]^5 with 500-call budget."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, fx, max_calls, rng, deadline):
        """Run Nelder-Mead from x0 with at most max_calls evaluations.
        Returns (best_x, best_fx, calls_made)."""
        n = self.dim
        # Build simplex
        simplex = [x0.copy()]
        fvals = [fx]
        for i in range(n):
            xi = x0.copy()
            # Standard NM initialization: stretch each vertex along axis i
            if abs(x0[i]) > 1e-10:
                xi[i] += 0.05 * abs(x0[i])
            else:
                xi[i] += 0.05
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            fvals.append(func(xi))

        calls = n + 1  # already used n+1 evaluations (x0 + n others)
        if calls >= max_calls:
            idx = np.argmin(fvals)
            return simplex[idx], fvals[idx], calls

        # NM parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        for _ in range(max_calls - calls):
            if time.time() > deadline:
                break
            # Sort
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            x_worst = simplex[-1]
            f_worst = fvals[-1]
            x_best = simplex[0]
            f_best = fvals[0]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_ref = centroid + alpha * (centroid - x_worst)
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls += 1

            if f_ref < f_best:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls += 1
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                x_con = centroid + rho * (x_worst - centroid)
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls += 1
                if f_con < f_worst:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                    if calls >= max_calls:
                        break

        # Find best
        idx = np.argmin(fvals)
        return simplex[idx], fvals[idx], calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Generate deterministic starting points spread across [-4, 4]^5
        # Use a stratified grid + random perturbations
        n_starts = 10
        calls_per_start = budget // n_starts  # 50 calls each

        # Create diverse starting points
        start_points = []
        # Include origin and axis-aligned points
        start_points.append(np.zeros(dim))
        for i in range(dim):
            p = np.zeros(dim)
            p[i] = 2.0
            start_points.append(p)
        # Random points in [-4, 4]^5
        while len(start_points) < n_starts:
            p = rng.uniform(-4, 4, dim)
            start_points.append(p)
        start_points = start_points[:n_starts]

        best_x = None
        best_fx = np.inf
        total_calls = 0

        for si, sp in enumerate(start_points):
            if total_calls >= budget:
                break
            if time.time() > deadline:
                break

            remaining = budget - total_calls
            max_calls_this = min(calls_per_start, remaining)
            if max_calls_this < 2:
                break

            fx = func(np.clip(sp, -5, 5))
            total_calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = np.clip(sp, -5, 5).copy()

            if total_calls >= budget:
                break
            if time.time() > deadline:
                break

            # Run NM from this start
            x_res, f_res, calls_made = self._nelder_mead(
                func, np.clip(sp, -5, 5), fx, max_calls_this, rng, deadline
            )
            total_calls += calls_made - 1  # fx already counted
            if f_res < best_fx:
                best_fx = f_res
                best_x = x_res.copy()

        # If we haven't used all budget and time remains, do a final polish
        if best_x is not None and total_calls < budget and time.time() < deadline:
            remaining = budget - total_calls
            if remaining >= 2:
                x_res, f_res, calls_made = self._nelder_mead(
                    func, best_x.copy(), best_fx, remaining, rng, deadline
                )
                total_calls += calls_made
                if f_res < best_fx:
                    best_fx = f_res
                    best_x = x_res.copy()

        if best_x is None:
            # Fallback: should not happen
            best_x = np.zeros(dim)
            best_fx = func(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
