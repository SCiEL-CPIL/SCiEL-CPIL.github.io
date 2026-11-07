"""Optimized Sphere (f1) minimizer with Nelder-Mead + adaptive restarts."""
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive restarts and time guard for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, budget_left, deadline, rng):
        """Run Nelder-Mead optimization with a budget cap and deadline."""
        n = self.dim
        # Build initial simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0
        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1, i] += 0.5

        # Evaluate initial simplex
        fvals = np.array([func(simplex[i]) for i in range(n + 1)])
        calls = n + 1  # Count initial evaluations

        # If we exceeded budget just from initialization, return immediately
        if calls > budget_left:
            return simplex[np.argmin(fvals)].copy(), fvals.min(), calls

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        best_x = simplex[np.argmin(fvals)].copy()
        best_f = fvals.min()

        while calls < budget_left and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            # Check convergence
            if fvals[-1] - fvals[0] < 1e-10:
                break

            # Centroid of all but worst
            centroid = simplex[:-1].mean(axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            if time.time() >= deadline:
                break

            if fr < fvals[0]:
                # Try expansion
                if calls + 1 > budget_left:
                    # Not enough budget for expansion, just accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if time.time() >= deadline:
                        break
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
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                
                if calls + 1 > budget_left:
                    # Not enough budget for contraction, accept reflection if it's better than worst? 
                    # Or just break? If reflection was worse than worst, we didn't replace it yet.
                    # In this branch, fr >= fvals[-2]. 
                    # If we can't contract, we should probably just keep the current simplex (don't replace worst with reflection if reflection is bad).
                    # But we already evaluated reflection.
                    # If fr < fvals[-1], reflection is better than worst. Replace worst with reflection.
                    if fr < fvals[-1]:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    break # No budget for further steps in this iteration
                else:
                    fc = func(xc)
                    calls += 1
                    if time.time() >= deadline:
                        break
                    if fc <= min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        # Shrink requires n calls. Check budget.
                        if calls + n > budget_left:
                            # Not enough budget for full shrink. 
                            # Just accept reflection if it's better than worst?
                            if fr < fvals[-1]:
                                simplex[-1] = xr
                                fvals[-1] = fr
                            break
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if time.time() >= deadline or calls >= budget_left:
                                break
                        if time.time() >= deadline or calls >= budget_left:
                            break

            # Track best
            if fvals.min() < best_f:
                best_f = fvals.min()
                best_x = simplex[np.argmin(fvals)].copy()

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Phase 1: Use several random starting points with Nelder-Mead
        # Allocate budget across restarts
        # We use a dynamic allocation: each restart gets the remaining budget
        # but we cap the number of restarts to ensure we don't waste too much on bad starts.
        # Given budget=500, dim=5, 5 restarts of ~100 calls is reasonable if strictly enforced.
        n_restarts = 5

        for r in range(n_restarts):
            if time.time() >= deadline:
                break
            remaining_budget = budget - calls_used
            if remaining_budget <= 0:
                break

            # Choose starting point
            if r == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4, 4, dim)

            # Pass the full remaining budget to allow Nelder-Mead to use it all
            # if it performs well, or stop if it hits the cap.
            # We don't restrict it to a fixed chunk anymore, just the total remaining.
            x_best, f_best, calls = self._nelder_mead(
                func, x0, remaining_budget, deadline, rng
            )
            calls_used += calls

            if f_best < best_f:
                best_f = f_best
                best_x = x_best

            # If we've converged very well, stop early
            if best_f < 1e-6:
                break

            # Optional: If we haven't used much budget and the result is poor,
            # we might want to stop and try a different strategy, but for now
            # just continue to next restart if budget allows.
            # To prevent 5 full runs if the first one is bad but uses all budget,
            # the loop condition `remaining_budget <= 0` handles it.
            # However, if the first restart uses 100 calls and finds a local min,
            # we have 400 left. The next restart will get 400.
            # This is fine.

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
