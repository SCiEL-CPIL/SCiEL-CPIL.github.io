"""Rewritten optimizer: Nelder-Mead simplex with multi-start and adaptive reset."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free optimizer using Nelder-Mead simplex with multi-start."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0

    def __call__(self, func):
        self._call_count = 0
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        # Wrap func to count calls and enforce budget
        def counted_func(x):
            self._call_count += 1
            if self._call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            return func(x)

        # Nelder-Mead implementation
        def nelder_mead(func, x0, max_calls, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
            """Nelder-Mead simplex method."""
            n = len(x0)
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [func(x0)]
            calls_used = 1

            # Create initial simplex around x0
            for i in range(n):
                xi = x0.copy()
                # Choose step size based on initial scale
                step = 0.5 if abs(x0[i]) < 1e-10 else 0.5 * abs(x0[i])
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
                fvals.append(func(xi))
                calls_used += 1

            # Sort simplex by function value
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            best_x = simplex[0].copy()
            best_f = fvals[0]

            while calls_used < max_calls:
                # Check time
                if time.time() > deadline - 5:
                    break

                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_used += 1

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_used += 1

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
                        xc = centroid + sigma * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - sigma * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_used += 1

                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls_used += 1
                        if calls_used >= max_calls:
                            break

                # Re-sort
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

                # Check convergence
                if max(fvals) - min(fvals) < 1e-10:
                    break

            return best_x, best_f, calls_used

        # Multi-start strategy
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # Phase 1: Coarse grid search to find good starting points
        # Use a small number of random starts + grid points
        n_starts = 8
        starts = []

        # Add some structured starts
        for _ in range(n_starts):
            x = rng.uniform(-5, 5, self.dim)
            starts.append(x)

        # Add center and some axis-aligned points
        starts.append(np.zeros(self.dim))
        for i in range(self.dim):
            for sign in [-1, 1]:
                x = np.zeros(self.dim)
                x[i] = sign * 3.0
                starts.append(x)

        # Deduplicate and limit starts
        seen = set()
        unique_starts = []
        for s in starts:
            key = tuple(np.round(s, 2))
            if key not in seen:
                seen.add(key)
                unique_starts.append(s)

        # Budget allocation: give more budget to promising starts
        # First, evaluate all starts
        start_results = []
        for s in unique_starts:
            if self._call_count >= self.budget:
                break
            if time.time() > deadline - 10:
                break
            fs = counted_func(s)
            start_results.append((fs, s))

        if not start_results:
            # Fallback: just do one evaluation
            x0 = np.zeros(self.dim)
            f0 = counted_func(x0)
            return x0, f0

        # Sort starts by function value
        start_results.sort(key=lambda x: x[0])

        # Allocate remaining budget to top starts
        remaining_budget = self.budget - self._call_count
        if remaining_budget <= 0:
            # Just return the best start
            best_f, best_x = start_results[0]
            return best_x, best_f

        # Use top few starts for Nelder-Mead refinement
        n_refine = min(3, len(start_results))
        budget_per_refine = remaining_budget // n_refine

        for i in range(n_refine):
            if self._call_count >= self.budget:
                break
            if time.time() > deadline - 10:
                break

            fs, xs = start_results[i]
            if fs < best_f:
                best_f = fs
                best_x = xs.copy()

            # Run Nelder-Mead from this start
            # Reserve some calls for the NM method
            nm_budget = min(budget_per_refine, self.budget - self._call_count)
            if nm_budget < 2 * self.dim + 1:
                continue

            x_opt, f_opt, _ = nelder_mead(
                counted_func, xs, nm_budget,
                alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5
            )

            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # Final polish: if we have budget left, do a fine NM around best_x
        if best_x is not None and self._call_count < self.budget:
            remaining = self.budget - self._call_count
            if remaining > 2 * self.dim + 1:
                # Fine-tune around best_x with smaller step
                x_opt, f_opt, _ = nelder_mead(
                    counted_func, best_x, remaining,
                    alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5
                )
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt.copy()

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = counted_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
