"""Fresh design: multi-start Nelder-Mead with coordinate polish for BBOB f20 (Schwefel)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with coordinate refinement for Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, max_calls, deadline, call_counter):
        """Run Nelder-Mead from x0, respecting max_calls and deadline."""
        dim = self.dim
        n = dim + 1
        # Build initial simplex
        simplex = np.zeros((n, dim))
        simplex[0] = x0
        for i in range(1, n):
            simplex[i] = x0.copy()
            simplex[i, i - 1] += 0.5

        # Evaluate initial simplex
        fvals = np.zeros(n)
        for i in range(n):
            if call_counter[0] >= max_calls or time.time() >= deadline:
                break
            x_i = np.clip(simplex[i], -5, 5)
            fvals[i] = func(x_i)
            call_counter[0] += 1
            simplex[i] = x_i

        if call_counter[0] < n:
            # Not enough calls to evaluate full simplex
            best_idx = np.argmin(fvals[:call_counter[0]])
            return simplex[best_idx], fvals[best_idx]

        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]

        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        while call_counter[0] < max_calls and time.time() < deadline:
            # Check for convergence
            if np.max(fvals) - np.min(fvals) < 1e-8:
                break

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            call_counter[0] += 1
            if call_counter[0] >= max_calls or time.time() >= deadline:
                if fr < fvals[-1]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                break

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                call_counter[0] += 1
                if call_counter[0] >= max_calls or time.time() >= deadline:
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    elif fr < fvals[-1]:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    break
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
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                call_counter[0] += 1
                if call_counter[0] >= max_calls or time.time() >= deadline:
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    break
                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if time.time() >= deadline:
                            break
                        fvals[i] = func(simplex[i])
                        call_counter[0] += 1
                    break

            # Re-sort
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]

    def _coordinate_polish(self, func, x, max_calls, deadline, call_counter):
        """Simple coordinate descent polish."""
        dim = self.dim
        x = x.copy()
        fx = func(x)
        call_counter[0] += 1
        if call_counter[0] >= max_calls or time.time() >= deadline:
            return x, fx

        step = 1.0
        improved = True
        while improved and call_counter[0] < max_calls and time.time() < deadline:
            improved = False
            for d in range(dim):
                for sign in (1, -1):
                    if call_counter[0] >= max_calls or time.time() >= deadline:
                        break
                    trial = x.copy()
                    trial[d] = np.clip(trial[d] + sign * step, -5, 5)
                    ft = func(trial)
                    call_counter[0] += 1
                    if ft < fx:
                        x = trial
                        fx = ft
                        improved = True
            step *= 0.5
            if step < 1e-6:
                break
        return x, fx

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 330s kill

        # Phase 1: Multi-start Nelder-Mead
        # Use ~70% of budget for NM, rest for polish
        nm_budget = int(budget * 0.7)
        polish_budget = budget - nm_budget

        # Number of restarts: use budget to determine
        # Each NM needs at least (dim+1) + some iterations
        # dim=5, so simplex has 6 points. Let's do ~4-6 restarts
        n_restarts = 5
        calls_per_nm = nm_budget // n_restarts

        call_counter = [0]
        best_x = None
        best_f = np.inf

        # Generate diverse starting points
        rng = np.random.default_rng(42)
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Random points
        for _ in range(n_restarts - 1):
            starts.append(rng.uniform(-4, 4, dim))

        for s in starts:
            if call_counter[0] >= nm_budget or time.time() >= deadline:
                break
            remaining = nm_budget - call_counter[0]
            x_best, f_best = self._nelder_mead(func, s, remaining, deadline, call_counter)
            if f_best < best_f:
                best_f = f_best
                best_x = x_best

        # Phase 2: Coordinate polish from best point
        if best_x is not None and call_counter[0] < budget and time.time() < deadline:
            remaining = budget - call_counter[0]
            best_x, best_f = self._coordinate_polish(func, best_x, remaining, deadline, call_counter)

        return best_x, best_f
# EVOLVE-BLOCK-END
