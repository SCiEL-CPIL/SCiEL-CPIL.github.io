"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead simplex search with periodic restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls = 0
        max_calls = self.budget

        # We'll use a budget of function calls.
        # Each Nelder-Mead iteration uses up to dim+1 = 6 function evaluations (worst case: reflect, expand, contract, shrink)
        # But we need to be careful. Let's do it more manually or track.

        # Strategy: run Nelder-Mead for a few hundred calls, if it plateaus, restart with a different initial simplex.
        # Since dim=5, simplex has 6 vertices.

        def nm_search(x0, budget_remaining):
            nonlocal calls, best_x, best_f
            dim = self.dim
            # Initialize simplex
            x0 = np.array(x0, dtype=float)
            simplex = [x0.copy()]
            # Add vertices
            for i in range(dim):
                v = x0.copy()
                step = 0.25 if x0[i] + 0.25 <= 5 else -0.25
                v[i] += step
                simplex.append(v)

            # Evaluate simplex
            fvals = []
            for v in simplex:
                v_clipped = np.clip(v, -5, 5)
                f = func(v_clipped)
                calls += 1
                fvals.append(f)
                if f < best_f:
                    best_f = f
                    best_x = v_clipped.copy()

            # Nelder-Mead iterations
            # We have budget_remaining - (dim+1) calls left for this local search
            # Each iteration can use up to 4 function calls (reflect, expand, contract, shrink)
            # Let's estimate ~2 calls per iteration on average

            n_iter_possible = max(0, budget_remaining - (dim + 1))
            # Be conservative: assume 2 calls per iteration
            n_iter = min(n_iter_possible // 2, 500)

            alpha, gamma, rho, sigma_ = 1.0, 2.0, 0.5, 0.5

            for _ in range(n_iter):
                if calls >= max_calls:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if calls >= max_calls:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fr < fvals[0]:
                    # Try expansion
                    if calls >= max_calls:
                        simplex[-1] = xr
                        fvals[-1] = fr
                        break
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
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
                    if calls >= max_calls:
                        break
                    # Outside contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    if fc < fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        if calls >= max_calls:
                            break
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_ * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls >= max_calls:
                                break
                            fi = func(simplex[i])
                            calls += 1
                            if fi < best_f:
                                best_f = fi
                                best_x = simplex[i].copy()
                            fvals[i] = fi
                        # Re-sort
                        order = np.argsort(fvals)
                        simplex = [simplex[i] for i in order]
                        fvals = [fvals[i] for i in order]

            return simplex[0], fvals[0]

        # Phase 1: Initial Nelder-Mead
        x0 = rng.uniform(-4, 4, self.dim)
        nm_search(x0, self.budget // 2)

        # Phase 2: If we have budget left, do another Nelder-Mead from best_x with small perturbation
        if best_x is not None and calls < self.budget:
            x0 = best_x + rng.normal(0, 0.5, self.dim)
            x0 = np.clip(x0, -5, 5)
            nm_search(x0, self.budget - calls)

        # Phase 3: Local fine-tuning with coordinate descent on best_x
        if best_x is not None and calls < self.budget:
            x = best_x.copy()
            f = best_f
            step = 0.1
            while calls < self.budget and step > 1e-6:
                improved = False
                for i in range(self.dim):
                    for direction in [1, -1]:
                        if calls >= self.budget:
                            break
                        x_try = x.copy()
                        x_try[i] += direction * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        calls += 1
                        if f_try < f:
                            x = x_try
                            f = f_try
                            improved = True
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()
                step *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
