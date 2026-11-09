import numpy as np
import time


class Optimizer:
    """Manual Nelder-Mead with (1+1)-ES refinement for Rosenbrock-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 280.0  # Safety margin before 300s kill

        def eval_func(x):
            """Evaluate with budget/time tracking."""
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        def nelder_mead_manual(x0, max_calls):
            """Manual Nelder-Mead with explicit budget control."""
            dim = self.dim
            n = dim + 1
            simplex = np.zeros((n, dim))
            simplex[0] = x0.copy()
            for i in range(1, n):
                simplex[i] = x0.copy()
                simplex[i, i - 1] += 0.5

            fvals = np.zeros(n)
            calls = 0
            for i in range(n):
                if calls >= max_calls or time.time() > self.deadline:
                    break
                fvals[i] = eval_func(simplex[i])
                calls += 1

            if calls < n:
                return

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

            while calls < max_calls and time.time() < self.deadline:
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                if np.abs(fvals[-1] - fvals[0]) < 1e-12:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                if calls >= max_calls or time.time() > self.deadline:
                    break
                fr = eval_func(xr)
                calls += 1

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    if calls >= max_calls or time.time() > self.deadline:
                        simplex[-1] = xr
                        fvals[-1] = fr
                        break
                    fe = eval_func(xe)
                    calls += 1
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
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (xr - centroid)
                    xc = np.clip(xc, -5, 5)
                    if calls >= max_calls or time.time() > self.deadline:
                        break
                    fc = eval_func(xc)
                    calls += 1
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        if calls >= max_calls or time.time() > self.deadline:
                            break
                        fvals[i] = eval_func(simplex[i])
                        calls += 1

        def es_refine(x_center, max_calls):
            """(1+1)-ES refinement with adaptive step size."""
            if max_calls < 5 or time.time() > self.deadline:
                return

            rng = np.random.default_rng(12345)
            x = x_center.copy()
            fx = self.best_f
            sigma = 0.5

            for _ in range(max_calls):
                if self.calls_made >= self.budget or time.time() > self.deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = eval_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.8

                if sigma < 1e-7:
                    break

        # Starting points: diverse but informed by Rosenbrock structure
        rng = np.random.default_rng(42)
        starts = [
            np.ones(self.dim),           # Standard Rosenbrock min
            np.zeros(self.dim),          # Origin
            np.full(self.dim, 2.0),      # Offset
        ]
        # Add random points in likely region and full box
        for _ in range(3):
            starts.append(rng.uniform(-2, 4, self.dim))
        for _ in range(2):
            starts.append(rng.uniform(-5, 5, self.dim))

        # Remove duplicates
        unique_starts = []
        for s in starts:
            if not any(np.allclose(s, u, atol=1e-5) for u in unique_starts):
                unique_starts.append(s)

        # Phase 1: Manual Nelder-Mead on multiple starts
        # Reserve ~20% budget for ES refinement
        nm_budget = int(self.budget * 0.8)
        es_budget = self.budget - nm_budget

        for x0 in unique_starts:
            if self.calls_made >= nm_budget or time.time() > self.deadline:
                break

            x0 = np.clip(x0, -5, 5)
            remaining = nm_budget - self.calls_made
            if remaining < 15:
                break

            try:
                nelder_mead_manual(x0, remaining)
            except StopIteration:
                break
            except Exception:
                continue

        # Phase 2: (1+1)-ES refinement on the best point found
        if self.best_x is not None and self.calls_made < self.budget:
            remaining = self.budget - self.calls_made
            es_refine(self.best_x, remaining)

        # Fallback if no calls were made
        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            self.best_f = func(np.clip(x0, -5, 5))

        return self.best_x, self.best_f
