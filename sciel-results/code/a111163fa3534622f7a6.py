"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
import numpy as np
import time


class Optimizer:
    """Model-aware hybrid optimizer exploiting Rastrigin's known structure."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        calls = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # ------------------------------------------------------------------
        # Fast local polish: Nelder-Mead + coordinate descent.
        # ------------------------------------------------------------------
        def polish(func, x0, budget_left, deadline):
            nonlocal calls
            d = self.dim
            x = np.array(x0, dtype=float)
            x = clip(x)
            fx = func(x)
            calls += 1
            best_x = x.copy()
            best_f = fx

            # --- Nelder-Mead ---
            delta = 0.3
            simplex = [x.copy()]
            simplex_f = [fx]
            for i in range(d):
                xi = x.copy()
                xi[i] += delta
                xi = clip(xi)
                fi = func(xi)
                calls += 1
                simplex.append(xi)
                simplex_f.append(fi)

            # Keep NM light: bounded number of iterations
            max_nm_iter = min(30, max(5, budget_left // 4))
            alpha, gamma, rho = 1.0, 2.0, -0.5

            for _ in range(max_nm_iter):
                if calls >= budget_left or time.time() > deadline - 5.0:
                    break
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = clip(centroid + alpha * (centroid - simplex[-1]))
                fr = func(xr)
                calls += 1

                if fr < simplex_f[0]:
                    # expansion
                    xe = clip(centroid + gamma * (xr - centroid))
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # contraction
                    xc = clip(centroid + rho * (simplex[-1] - centroid))
                    fc = func(xc)
                    calls += 1
                    if fc < simplex[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # shrink
                        for i in range(1, d + 1):
                            simplex[i] = clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                            simplex_f[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget_left:
                                break
                        if calls >= budget_left:
                            break

                # track best
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()
                if calls >= budget_left:
                    break

            # --- Coordinate descent refinement ---
            step = 0.1
            for _ in range(10):
                if calls >= budget_left or time.time() > deadline - 5.0:
                    break
                improved = False
                for i in range(d):
                    for sign in (-1, 1):
                        xi = best_x.copy()
                        xi[i] += sign * step
                        xi = clip(xi)
                        fi = func(xi)
                        calls += 1
                        if fi < best_f:
                            best_f = fi
                            best_x = xi
                            improved = True
                        if calls >= budget_left:
                            break
                    if calls >= budget_left:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-5:
                        break
                else:
                    step = min(step * 1.5, 0.5)

            return best_x, best_f

        # ------------------------------------------------------------------
        # Global search: Halton sequence + integer neighborhood sampling.
        # ------------------------------------------------------------------
        def halton(dim, n, skip=1):
            """Generate n points from the Halton sequence in dim dimensions."""
            points = np.zeros((n, dim))
            for d_i in range(dim):
                base = d_i + 2
                for j in range(n):
                    i = j + skip
                    f = 1.0
                    r = 0.0
                    while i > 0:
                        f /= base
                        r += f * (i % base)
                        i //= base
                    points[j, d_i] = r
            return points

        candidates = []  # list of (x, f)

        # Phase 1: Halton sequence for systematic coverage (60% of budget)
        n_halton = int(budget * 0.6)
        halton_points = halton(dim, n_halton, skip=10)
        halton_points = halton_points * 10.0 - 5.0  # map to [-5, 5]

        for i in range(n_halton):
            if calls >= budget or time.time() > deadline - 5.0:
                break
            x = clip(halton_points[i])
            fx = func(x)
            calls += 1
            candidates.append((x, fx))

        # Phase 2: Sample near integer coordinates (Rastrigin local minima)
        # The global minimum is at x* (hidden shift), local minima at integers.
        n_int = int(budget * 0.15)
        for _ in range(n_int):
            if calls >= budget or time.time() > deadline - 5.0:
                break
            x = rng.integers(-4, 5, dim).astype(float) + rng.uniform(-0.5, 0.5, dim)
            x = clip(x)
            fx = func(x)
            calls += 1
            candidates.append((x, fx))

        # Phase 3: Random restarts to fill gaps
        n_random = int(budget * 0.15)
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline - 5.0:
                break
            x = rng.uniform(-4.8, 4.8, dim)
            x = clip(x)
            fx = func(x)
            calls += 1
            candidates.append((x, fx))

        # Sort by f-value
        candidates.sort(key=lambda t: t[1])

        # ------------------------------------------------------------------
        # Phase 4: Polish top candidates with remaining budget.
        # ------------------------------------------------------------------
        # Allocate polish budget to top K candidates
        n_polish = min(8, len(candidates))
        
        best_x_global = candidates[0][0].copy()
        best_f_global = candidates[0][1]

        # Give more budget to better candidates
        for i in range(n_polish):
            if calls >= budget or time.time() > deadline - 5.0:
                break
            x0 = candidates[i][0]
            remaining = budget - calls
            if remaining < 15:
                break
            # Give more budget to better candidates
            if i == 0:
                pb = min(remaining, int(remaining * 0.4))
            elif i == 1:
                pb = min(remaining, int(remaining * 0.25))
            elif i == 2:
                pb = min(remaining, int(remaining * 0.2))
            else:
                pb = min(remaining, int(remaining * 0.1))
            pb = max(15, pb)

            px, pf = polish(func, x0, pb, deadline)
            if pf < best_f_global:
                best_f_global = pf
                best_x_global = px

        return best_x_global, best_f_global
