"""Optimizer for BBOB f22 (Gallagher 21 peaks) with multi-start Nelder-Mead + ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with ES fine-tuning and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 280.0  # safety margin before 300s limit
        calls_made = 0
        budget = self.budget
        dim = self.dim

        def eval_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return float('inf')
            calls_made += 1
            val = func(x)
            return val

        rng = np.random.default_rng(42)

        # Multi-start: use several initial points to explore different basins
        # Strategy: combine random points, center, and grid-like points
        n_starts = 15
        starts = []

        # Center
        starts.append(np.zeros(dim))

        # Random points
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        # Shuffle starts
        order = rng.permutation(n_starts)
        starts = [starts[i] for i in order]

        best_x = None
        best_f = float('inf')

        # Phase 1: Nelder-Mead from each start with limited budget per start
        nm_budget_per_start = max(20, (budget // 2) // n_starts)
        remaining_budget = budget

        for start in starts:
            if calls_made >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            if remaining_budget < 10:
                break

            # Run Nelder-Mead
            nm_result = self._nelder_mead(
                eval_func, start, dim,
                max_iter=min(nm_budget_per_start, remaining_budget),
                deadline=deadline
            )
            remaining_budget = budget - calls_made

            if nm_result is not None:
                x_nm, f_nm = nm_result
                if f_nm < best_f:
                    best_f = f_nm
                    best_x = x_nm.copy()

        # Phase 2: ES refinement around the best point
        if best_x is not None and calls_made < budget and time.time() < deadline - 2.0:
            es_result = self._es_refine(
                eval_func, best_x, best_f, dim,
                budget_remaining=budget - calls_made,
                deadline=deadline
            )
            if es_result is not None:
                x_es, f_es = es_result
                if f_es < best_f:
                    best_f = f_es
                    best_x = x_es

        # Phase 3: If we have budget left, try a few more random restarts
        if calls_made < budget and time.time() < deadline - 2.0:
            for _ in range(5):
                if calls_made >= budget or time.time() > deadline - 2.0:
                    break
                x_rand = rng.uniform(-4.5, 4.5, dim)
                f_rand = eval_func(x_rand)
                if f_rand < best_f:
                    best_f = f_rand
                    best_x = x_rand.copy()
                    # Quick NM from this new best
                    if calls_made < budget - 10 and time.time() < deadline - 5.0:
                        nm_result = self._nelder_mead(
                            eval_func, best_x, dim,
                            max_iter=min(50, budget - calls_made),
                            deadline=deadline
                        )
                        if nm_result is not None:
                            x_nm, f_nm = nm_result
                            if f_nm < best_f:
                                best_f = f_nm
                                best_x = x_nm.copy()

        if best_x is None:
            # Fallback: return a point we've evaluated
            best_x = np.zeros(dim)
            best_f = eval_func(best_x)

        return best_x, best_f

    def _nelder_mead(self, eval_func, x0, dim, max_iter, deadline):
        """Nelder-Mead simplex method with adaptive step size."""
        x = np.array(x0, dtype=float)
        f0 = eval_func(x)

        if not np.isfinite(f0):
            return None

        # Build initial simplex
        simplex = [x.copy()]
        simplex_f = [f0]

        # Initial step: scale based on dimension
        base_step = 0.5
        for i in range(dim):
            xi = x.copy()
            xi[i] += base_step
            simplex.append(xi)
            simplex_f.append(eval_func(xi))

        # Sort simplex
        order = np.argsort(simplex_f)
        simplex = [simplex[i] for i in order]
        simplex_f = [simplex_f[i] for i in order]

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        min_f = simplex_f[0]
        best_x = simplex[0].copy()

        for _ in range(max_iter):
            if time.time() > deadline:
                break

            # Compute centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = eval_func(xr)

            if fr < simplex_f[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = eval_func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = xr
                simplex_f[-1] = fr
            else:
                # Contraction
                if fr < simplex_f[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = eval_func(xc)

                if fc < min(simplex_f[-1], fr):
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = eval_func(simplex[i])

            # Sort
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            # Track best
            if simplex_f[0] < min_f:
                min_f = simplex_f[0]
                best_x = simplex[0].copy()

            # Convergence check
            if len(simplex_f) > 1 and abs(simplex_f[0] - simplex_f[-1]) < 1e-10:
                break

        return best_x, min_f

    def _es_refine(self, eval_func, x0, f0, dim, budget_remaining, deadline):
        """(1+1)-ES for fine-tuning."""
        x = np.array(x0, dtype=float)
        fx = f0
        sigma = 0.1  # Start with small step for refinement

        for _ in range(budget_remaining):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * np.random.default_rng(hash((x, sigma)) % (2**32)).standard_normal(dim), -5, 5)
            fy = eval_func(y)

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.9

            if sigma < 1e-8:
                break

        return x, fx
# EVOLVE-BLOCK-END
