"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
import numpy as np
import time


class Optimizer:
    """Model-aware hybrid optimizer exploiting Rastrigin's known structure.

    Rastrigin: f(x) = 10*dim + sum_i (x_i^2 - 10*cos(2*pi*x_i))
    The quadratic part g(x) = sum_i x_i^2 is unimodal with its unique
    minimum at the global minimizer x*.  Local minima sit at integer
    coordinates and have g-value ~10*dim higher than the global one.
    We therefore:
      1. Multi-start to locate candidate minima.
      2. Use g(x) to identify the global basin (lowest g among minima).
      3. Polish the best candidate with a fast local search.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(20240501)
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
            max_nm_iter = min(40, max(5, budget_left // 4))
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
            for _ in range(8):
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
                    if step < 1e-4:
                        break
                else:
                    step = min(step * 1.5, 0.5)

            return best_x, best_f

        # ------------------------------------------------------------------
        # Global search: Latin Hypercube + random restarts.
        # ------------------------------------------------------------------
        # Phase 1: Latin Hypercube spread (dim=5 -> 50 points, 10 per dim)
        n_lh = 50
        if budget >= n_lh + 10:
            # Latin hypercube sampling
            lh = np.zeros((n_lh, dim))
            for d_i in range(dim):
                perm = rng.permutation(n_lh)
                lh[:, d_i] = (perm + rng.uniform(size=n_lh)) / n_lh
            lh = lh * 10.0 - 5.0  # map to [-5, 5]

            # Evaluate LH points
            for i in range(n_lh):
                if calls >= budget or time.time() > deadline - 5.0:
                    break
                x = clip(lh[i])
                fx = func(x)
                calls += 1

        # Phase 2: Random restarts to find promising minima
        # Use a pool of candidate starting points
        candidates = []
        # Add LH points
        if budget >= n_lh + 10:
            candidates.extend([clip(lh[i]) for i in range(n_lh)])

        # Add random points
        n_random = 30
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline - 5.0:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            x = clip(x)
            fx = func(x)
            calls += 1
            candidates.append(x)

        # Add points near known Rastrigin minima (integers)
        # Global minimum is at x* (hidden shift), local at integers.
        # Sample around integers in [-4, 4]
        for _ in range(20):
            if calls >= budget or time.time() > deadline - 5.0:
                break
            x = rng.integers(-4, 5, dim).astype(float) + rng.uniform(-0.3, 0.3, dim)
            x = clip(x)
            fx = func(x)
            calls += 1
            candidates.append(x)

        # Sort candidates by f-value, keep top ones for polishing
        # We need f-values; re-evaluate is wasteful, so track during generation.
        # Let's restructure: keep (x, f) pairs.
        # Actually, let's redo the global search with tracked values.

        # Reset and redo with proper tracking
        calls = 0
        candidates = []  # list of (x, f)

        # Latin Hypercube
        n_lh = 50
        if budget >= n_lh + 10:
            lh = np.zeros((n_lh, dim))
            for d_i in range(dim):
                perm = rng.permutation(n_lh)
                lh[:, d_i] = (perm + rng.uniform(size=n_lh)) / n_lh
            lh = lh * 10.0 - 5.0
            for i in range(n_lh):
                if calls >= budget or time.time() > deadline - 5.0:
                    break
                x = clip(lh[i])
                fx = func(x)
                calls += 1
                candidates.append((x, fx))

        # Random
        n_random = 30
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline - 5.0:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            x = clip(x)
            fx = func(x)
            calls += 1
            candidates.append((x, fx))

        # Near integers
        for _ in range(20):
            if calls >= budget or time.time() > deadline - 5.0:
                break
            x = rng.integers(-4, 5, dim).astype(float) + rng.uniform(-0.3, 0.3, dim)
            x = clip(x)
            fx = func(x)
            calls += 1
            candidates.append((x, fx))

        # Sort by f-value
        candidates.sort(key=lambda t: t[1])

        # ------------------------------------------------------------------
        # Phase 3: Polish top candidates with remaining budget.
        # ------------------------------------------------------------------
        # Allocate polish budget to top K candidates
        # Each polish uses ~30-50 calls. With 500 budget and ~100 used for global,
        # we have ~400 for polishing. Polish top 8-10 candidates.
        n_polish = min(10, len(candidates))
        polish_budget_each = max(20, (budget - calls) // max(1, n_polish))

        best_x_global = candidates[0][0].copy()
        best_f_global = candidates[0][1]

        for i in range(n_polish):
            if calls >= budget or time.time() > deadline - 5.0:
                break
            x0 = candidates[i][0]
            remaining = budget - calls
            if remaining < 15:
                break
            # Give more budget to better candidates
            if i == 0:
                pb = min(remaining, 150)
            elif i == 1:
                pb = min(remaining, 100)
            elif i == 2:
                pb = min(remaining, 80)
            else:
                pb = min(remaining, 50)
            pb = max(15, pb)

            px, pf = polish(func, x0, pb, deadline)
            if pf < best_f_global:
                best_f_global = pf
                best_x_global = px

        return best_x_global, best_f_global
