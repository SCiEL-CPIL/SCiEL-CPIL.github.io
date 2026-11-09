import numpy as np
import time

class Optimizer:
    """Improved optimizer: Nelder-Mead + CMA-ES with adaptive strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin

        calls = [0]

        def safe_func(x):
            if calls[0] >= budget:
                raise StopIteration
            calls[0] += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # ---- Phase 1: Nelder-Mead with multiple starts ----
        starts = [
            np.array([0.0] * dim),
            np.array([2.0] * dim),
            np.array([-2.0] * dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
        ]

        for s in starts:
            if calls[0] >= budget or time.time() > deadline:
                break
            x0 = np.clip(s, -5, 5)
            try:
                fx0 = safe_func(x0)
            except StopIteration:
                break
            update_best(x0, fx0)

            # Build simplex
            simplex = [x0.copy()]
            fvals = [fx0]
            for d in range(dim):
                if calls[0] >= budget or time.time() > deadline:
                    break
                x_d = x0.copy()
                x_d[d] += 0.5
                x_d = np.clip(x_d, -5, 5)
                try:
                    f_d = safe_func(x_d)
                except StopIteration:
                    break
                simplex.append(x_d)
                fvals.append(f_d)
                update_best(x_d, f_d)

            if len(simplex) < dim + 1:
                continue

            # Nelder-Mead iterations
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            for _ in range(50):
                if calls[0] >= budget or time.time() > deadline:
                    break
                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                try:
                    f_r = safe_func(x_r)
                except StopIteration:
                    break
                update_best(x_r, f_r)

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    try:
                        f_e = safe_func(x_e)
                    except StopIteration:
                        break
                    update_best(x_e, f_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    try:
                        f_c = safe_func(x_c)
                    except StopIteration:
                        break
                    update_best(x_c, f_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            try:
                                f_s = safe_func(simplex[i])
                            except StopIteration:
                                break
                            fvals[i] = f_s
                            update_best(simplex[i], f_s)
                        if calls[0] >= budget or time.time() > deadline:
                            break

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-8:
                    break

        # ---- Phase 2: CMA-ES refinement ----
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            center = best_x.copy()
            cf = best_f
            sigma = 0.3
            lam = 6

            while calls[0] < budget and time.time() < deadline:
                improved_any = False
                for i in range(lam):
                    if calls[0] >= budget or time.time() > deadline:
                        break
                    child = np.clip(center + sigma * rng.standard_normal(dim), -5, 5)
                    try:
                        cf_c = safe_func(child)
                    except StopIteration:
                        break
                    if cf_c < cf:
                        cf = cf_c
                        center = child
                        improved_any = True
                    update_best(child, cf_c)

                if improved_any:
                    sigma *= 1.1
                else:
                    sigma *= 0.85

                if sigma < 1e-6:
                    break

        return best_x, best_f
