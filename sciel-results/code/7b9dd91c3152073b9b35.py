import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def f(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls += 1
            xc = np.clip(x, -5.0, 5.0)
            val = float(func(xc))
            if val < best_f:
                best_f = val
                best_x = xc.copy()
            return val

        # ---- Nelder-Mead (from inspiration 1, which scored 2.07) ----
        # Nelder-Mead adapts a simplex in the full space, which is well-suited
        # to the hidden rotation of f13. We implement it manually to control
        # the call budget precisely.
        def nelder_mead(x0, max_iter=60):
            n = dim
            # Build initial simplex: x0 plus x0 + delta * e_i
            delta = 2.0
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                xi[i] += delta
                simplex.append(xi)
            vals = [f(xi) for xi in simplex]

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break
                # Sort simplex by function value
                order = np.argsort(vals)
                simplex = [simplex[i] for i in order]
                vals = [vals[i] for i in order]

                # Centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_val = vals[-1]

                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                f_r = f(x_r)

                if f_r < vals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - worst)
                    f_e = f(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        vals[-1] = f_r
                elif f_r < vals[-2]:
                    simplex[-1] = x_r
                    vals[-1] = f_r
                else:
                    # Contraction
                    if f_r < worst_val:
                        x_c = centroid + rho * (centroid - worst)
                        f_c = f(x_c)
                        if f_c <= f_r:
                            simplex[-1] = x_c
                            vals[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                vals[i] = f(simplex[i])
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            vals[i] = f(simplex[i])

            # Return the best point in the final simplex
            best_idx = np.argmin(vals)
            return simplex[best_idx], vals[best_idx]

        # ---- (1+1)-ES refinement with adaptive step size ----
        def es(x, fx, iters):
            rng = np.random.default_rng(42)
            sigma = 0.3
            for _ in range(iters):
                if time.time() > deadline or calls >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = f(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                else:
                    sigma *= 0.85
            return x, fx

        # ---- Multi-start strategy ----
        # The key insight from the cell findings: multi-start random basin-finding
        # helps locate the hidden shifted minimum. We do a small number of
        # independent Nelder-Mead runs from different starting points, then
        # polish the best with ES.
        rng0 = np.random.default_rng(7)

        # First, a quick random scan to find a good starting region
        # Use a few random points to identify a promising area
        candidates = []
        for i in range(8):
            if time.time() > deadline or calls >= budget:
                break
            x_try = rng0.uniform(-4.0, 4.0, dim)
            fx_try = f(x_try)
            candidates.append((fx_try, x_try))

        # Sort by function value
        candidates.sort(key=lambda t: t[0])

        # Run Nelder-Mead from the best 2-3 starting points
        best_nm_x = None
        best_nm_f = np.inf

        for i in range(min(3, len(candidates))):
            if time.time() > deadline or calls >= budget:
                break
            x_start = candidates[i][1].copy()
            x_nm, fx_nm = nelder_mead(x_start, max_iter=50)
            if fx_nm < best_nm_f:
                best_nm_f = fx_nm
                best_nm_x = x_nm

        # Polish with ES
        if best_nm_x is not None:
            x_final, fx_final = es(best_nm_x, best_nm_f, 100)

            # If budget remains, do another round of ES with smaller steps
            if time.time() < deadline and calls < budget:
                x_final, fx_final = es(x_final, fx_final, 80)

        return best_x, best_f
