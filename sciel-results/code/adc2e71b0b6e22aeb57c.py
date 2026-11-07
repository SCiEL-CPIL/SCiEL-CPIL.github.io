"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead ensemble for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n_starts = 5
        dim = self.dim
        budget = self.budget

        # Generate space-filling initial points using Latin Hypercube sampling
        lhs_points = np.zeros((n_starts, dim))
        for d in range(dim):
            strata = np.array([
                rng.uniform(-5 + 10.0 * i / n_starts, -5 + 10.0 * (i + 1) / n_starts)
                for i in range(n_starts)
            ])
            perm = rng.permutation(n_starts)
            for i in range(n_starts):
                lhs_points[i, d] = strata[perm[i]]

        # Initialize best tracking
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Initialize simplexes
        simplex_starts = []
        for i in range(n_starts):
            x0 = lhs_points[i]
            fx0 = func(x0)
            calls_made += 1
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            # Build initial simplex around x0 with larger initial step
            simplex = [x0.copy()]
            simplex_fvals = [fx0]

            for d in range(dim):
                step = 0.5
                new_point = x0.copy()
                new_point[d] += step
                new_point = np.clip(new_point, -5, 5)
                if np.allclose(new_point, x0):
                    new_point = x0.copy()
                    new_point[d] -= step
                    new_point = np.clip(new_point, -5, 5)
                fnew = func(new_point)
                calls_made += 1
                if fnew < best_f:
                    best_f = fnew
                    best_x = new_point.copy()
                simplex.append(new_point)
                simplex_fvals.append(fnew)

            simplex_starts.append({
                'simplex': simplex,
                'fvals': simplex_fvals,
                'active': True
            })

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        remaining = budget - calls_made

        while remaining > 0 and any(s['active'] for s in simplex_starts):
            for si, s in enumerate(simplex_starts):
                if remaining <= 0:
                    break
                if not s['active']:
                    continue

                simplex = s['simplex']
                fvals = s['fvals']
                n = len(simplex)

                # Sort by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Compute centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_worst = simplex[-1]
                f_worst = fvals[-1]
                x_ref = centroid + alpha * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls_made += 1
                remaining -= 1
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()

                if f_ref < fvals[0]:
                    if remaining <= 0:
                        break
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls_made += 1
                    remaining -= 1
                    if f_exp < best_f:
                        best_f = f_exp
                        best_x = x_exp.copy()

                    if f_exp < f_ref:
                        x_new, f_new = x_exp, f_exp
                    else:
                        x_new, f_new = x_ref, f_ref
                elif f_ref < fvals[-2]:
                    x_new, f_new = x_ref, f_ref
                else:
                    if remaining <= 0:
                        break
                    if f_ref < f_worst:
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        x_con = centroid - rho * (x_worst - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls_made += 1
                    remaining -= 1
                    if f_con < best_f:
                        best_f = f_con
                        best_x = x_con.copy()

                    if f_con < min(f_ref, f_worst):
                        x_new, f_new = x_con, f_con
                    else:
                        if remaining <= 0:
                            break
                        x_shrunk = []
                        f_shrunk = []
                        for i in range(1, n):
                            x_shr = simplex[0] + sigma * (simplex[i] - simplex[0])
                            x_shr = np.clip(x_shr, -5, 5)
                            f_shr = func(x_shr)
                            calls_made += 1
                            remaining -= 1
                            if f_shr < best_f:
                                best_f = f_shr
                                best_x = x_shr.copy()
                            x_shrunk.append(x_shr)
                            f_shrunk.append(f_shr)
                        simplex = [simplex[0]] + x_shrunk
                        fvals = [fvals[0]] + f_shrunk
                        s['simplex'] = simplex
                        s['fvals'] = fvals
                        continue

                simplex[-1] = x_new
                fvals[-1] = f_new
                s['simplex'] = simplex
                s['fvals'] = fvals

                # Check convergence
                spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                if np.max(spread) < 1e-6:
                    s['active'] = False

        return best_x, best_f
# EVOLVE-BLOCK-END
