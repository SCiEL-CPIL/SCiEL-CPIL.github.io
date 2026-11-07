"""Rewritten optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with multi-start on Gallagher 21-peaks (f22)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, fx0, max_iter, rng, deadline, calls):
        """Run Nelder-Mead from x0. Returns (best_x, best_f, calls_used)."""
        n = self.dim
        # Build simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0
        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += 0.5

        fvals = np.zeros(n + 1)
        fvals[0] = fx0
        calls[0] += 1

        for i in range(1, n + 1):
            simplex[i] = np.clip(simplex[i], -5, 5)
            fvals[i] = func(simplex[i])
            calls[0] += 1

        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        best_x = x0.copy()
        best_f = fx0

        for _ in range(max_iter):
            if time.time() >= deadline:
                break
            if calls[0] >= self.budget - 5:
                break

            # Sort by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_ref = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
            f_ref = func(x_ref)
            calls[0] += 1

            if f_ref < fvals[0]:
                # Expansion
                x_exp = np.clip(centroid + gamma * (x_ref - centroid), -5, 5)
                f_exp = func(x_exp)
                calls[0] += 1
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
                if f_ref < fvals[-1]:
                    x_con = np.clip(centroid + rho * (x_ref - centroid), -5, 5)
                else:
                    x_con = np.clip(centroid - rho * (simplex[-1] - centroid), -5, 5)
                f_con = func(x_con)
                calls[0] += 1
                if f_con <= min(f_ref, fvals[-1]):
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5, 5)
                        fvals[i] = func(simplex[i])
                        calls[0] += 1
                    if time.time() >= deadline or calls[0] >= self.budget - 5:
                        break

        return best_x, best_f

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        calls = [0]

        def safe_func(x):
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            calls[0] += 1
            return float(func(x))

        best_x = None
        best_f = np.inf

        # Phase 1: Broad exploration with multiple random starts
        # Use Nelder-Mead from several starting points
        n_starts = 10
        for s in range(n_starts):
            if time.time() >= deadline:
                break
            if calls[0] >= self.budget - 20:
                break

            if s == 0:
                x0 = np.zeros(self.dim)
            elif s == 1:
                x0 = rng.uniform(-4, 4, self.dim)
            elif s == 2:
                x0 = rng.uniform(-4, 4, self.dim)
            else:
                # Perturb around previous best
                if best_x is not None:
                    x0 = np.clip(best_x + 1.5 * rng.standard_normal(self.dim), -5, 5)
                else:
                    x0 = rng.uniform(-4, 4, self.dim)

            x0 = np.clip(x0, -5, 5)
            fx0 = safe_func(x0)

            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            # Run Nelder-Mead with limited iterations
            remaining = self.budget - calls[0] - 10
            if remaining <= 0:
                break
            # Allocate roughly equal budget to each start, but allow more for later (exploit)
            max_iter = max(20, min(80, remaining // max(1, n_starts - s)))

            bx, bf = self._nelder_mead(safe_func, x0, fx0, max_iter, rng, deadline, calls)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        # Phase 2: Fine-tuning around the best point found
        if best_x is not None and calls[0] < self.budget - 5 and time.time() < deadline:
            # Run a tighter Nelder-Mead around the best point
            x_start = best_x.copy()
            fx_start = safe_func(x_start)
            if fx_start < best_f:
                best_f = fx_start
                best_x = x_start.copy()

            remaining = self.budget - calls[0] - 5
            if remaining > 10:
                # Smaller initial simplex for fine-tuning
                n = self.dim
                simplex = np.zeros((n + 1, n))
                simplex[0] = x_start
                for i in range(n):
                    simplex[i + 1] = x_start.copy()
                    simplex[i + 1][i] += 0.05

                fvals = np.zeros(n + 1)
                fvals[0] = fx_start
                calls[0] += 1

                for i in range(1, n + 1):
                    simplex[i] = np.clip(simplex[i], -5, 5)
                    fvals[i] = safe_func(simplex[i])
                    calls[0] += 1

                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5

                for _ in range(remaining):
                    if time.time() >= deadline:
                        break
                    if calls[0] >= self.budget - 2:
                        break

                    order = np.argsort(fvals)
                    simplex = simplex[order]
                    fvals = fvals[order]

                    if fvals[0] < best_f:
                        best_f = fvals[0]
                        best_x = simplex[0].copy()

                    centroid = np.mean(simplex[:-1], axis=0)
                    x_ref = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                    f_ref = safe_func(x_ref)

                    if f_ref < fvals[0]:
                        x_exp = np.clip(centroid + gamma * (x_ref - centroid), -5, 5)
                        f_exp = safe_func(x_exp)
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
                        if f_ref < fvals[-1]:
                            x_con = np.clip(centroid + rho * (x_ref - centroid), -5, 5)
                        else:
                            x_con = np.clip(centroid - rho * (simplex[-1] - centroid), -5, 5)
                        f_con = safe_func(x_con)
                        if f_con <= min(f_ref, fvals[-1]):
                            simplex[-1] = x_con
                            fvals[-1] = f_con
                        else:
                            for i in range(1, n + 1):
                                simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5, 5)
                                fvals[i] = safe_func(simplex[i])
                            if time.time() >= deadline or calls[0] >= self.budget - 2:
                                break

        return best_x, best_f
# EVOLVE-BLOCK-END
