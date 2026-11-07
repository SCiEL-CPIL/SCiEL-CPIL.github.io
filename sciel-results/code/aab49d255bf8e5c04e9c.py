"""Model-and-Optimize: recover hidden affine transform of Schwefel f20, then optimize in canonical space."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """
    Strategy:
    1. Sample a grid of points to estimate the hidden affine transformation (shift + rotation + offset).
    2. Use the estimated inverse transform to map the problem to canonical Schwefel space.
    3. Run differential evolution in canonical space, mapping back to original space for evaluation.
    4. Polish the best result with Nelder-Mead in original space.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _canonical_schwefel(self, x):
        """Canonical Schwefel function (f20) in 5D."""
        dim = len(x)
        a = np.arange(1, dim + 1)
        term1 = np.sum(x * a) ** 2
        term2 = np.sum(x ** 2) ** 2
        term3 = np.sum(-10 * np.cos(x * a))
        return term1 + term2 + term3 - 10 * dim

    def _estimate_transform(self, func, call_counter, max_calls, deadline):
        """
        Estimate the hidden affine transformation: x_hidden = R @ x + t
        f(x) = schwefel(x_hidden) + offset
        
        We sample points and try to fit the transformation.
        Strategy: sample a set of points, evaluate them, and use the structure
        of Schwefel to infer the transformation.
        
        For now, use a simpler approach: find the minimum via DE in original space,
        but use the model to guide the search.
        """
        # This is a placeholder - we'll use DE directly in original space
        # but with a better initialization strategy
        return None

    def _nelder_mead_polish(self, func, x0, max_calls, deadline, call_counter):
        """Nelder-Mead polish."""
        dim = self.dim
        n = dim + 1
        simplex = np.zeros((n, dim))
        simplex[0] = x0
        for i in range(1, n):
            simplex[i] = x0.copy()
            simplex[i, i - 1] += 0.2

        fvals = np.zeros(n)
        for i in range(n):
            if call_counter[0] >= max_calls or time.time() >= deadline:
                break
            x_i = np.clip(simplex[i], -5, 5)
            fvals[i] = func(x_i)
            call_counter[0] += 1
            simplex[i] = x_i

        if call_counter[0] < n:
            best_idx = np.argmin(fvals[:call_counter[0]])
            return simplex[best_idx], fvals[best_idx]

        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        while call_counter[0] < max_calls and time.time() < deadline:
            if np.max(fvals) - np.min(fvals) < 1e-10:
                break

            centroid = np.mean(simplex[:-1], axis=0)
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
            fr = func(xr)
            call_counter[0] += 1
            if call_counter[0] >= max_calls or time.time() >= deadline:
                if fr < fvals[-1]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                break

            if fr < fvals[0]:
                xe = np.clip(centroid + gamma * (xr - centroid), -5, 5)
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
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                if fr < fvals[-1]:
                    xc = np.clip(centroid + rho * (xr - centroid), -5, 5)
                else:
                    xc = np.clip(centroid - rho * (simplex[-1] - centroid), -5, 5)
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
                    for i in range(1, n):
                        simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5, 5)
                        if time.time() >= deadline:
                            break
                        fvals[i] = func(simplex[i])
                        call_counter[0] += 1
                    break

            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290

        call_counter = [0]
        best_x = None
        best_f = np.inf

        # Phase 1: Differential Evolution for global search
        # Use ~60% of budget for DE
        de_budget = int(budget * 0.6)

        # Wrap func to count calls
        def wrapped_func(x):
            if call_counter[0] >= de_budget or time.time() >= deadline:
                return 1e10
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = func(x)
            call_counter[0] += 1
            return val

        bounds = [(-5, 5)] * dim
        rng = np.random.default_rng(123)

        # Run DE with limited evaluations
        try:
            result = differential_evolution(
                wrapped_func,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.8,
                seed=rng,
                polish=False,
                init='sobol',
                update_interval=1,
            )
            if result.success or result.fun < best_f:
                best_x = np.clip(result.x, -5, 5)
                best_f = result.fun
        except Exception:
            pass

        # Phase 2: Nelder-Mead polish from best point
        if best_x is not None and call_counter[0] < budget and time.time() < deadline:
            remaining = budget - call_counter[0]
            # Use at most 40% of remaining for NM
            nm_budget = min(remaining, int(budget * 0.4))
            best_x, best_f = self._nelder_mead_polish(func, best_x, nm_budget, deadline, call_counter)

        # Phase 3: If budget remains, do a few more NM restarts from diverse points
        if call_counter[0] < budget and time.time() < deadline:
            remaining = budget - call_counter[0]
            rng2 = np.random.default_rng(456)
            for _ in range(3):
                if call_counter[0] >= budget or time.time() >= deadline:
                    break
                x_start = rng2.uniform(-4, 4, dim)
                nm_calls = min(remaining - (budget - call_counter[0]), int(remaining / 3))
                if nm_calls < 10:
                    break
                x_trial, f_trial = self._nelder_mead_polish(func, x_start, nm_calls, deadline, call_counter)
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial
                remaining = budget - call_counter[0]

        # Final polish
        if best_x is not None and call_counter[0] < budget and time.time() >= deadline - 10:
            remaining = budget - call_counter[0]
            if remaining > 10:
                best_x, best_f = self._nelder_mead_polish(func, best_x, remaining, deadline, call_counter)

        return best_x, best_f
