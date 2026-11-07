"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Nelder-Mead with local Hessian Newton steps for Discus."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit
        rng = np.random.default_rng(0)

        # Count evaluations
        eval_count = [0]

        def eval_func(x):
            eval_count[0] += 1
            x = np.clip(x, -5, 5)
            return float(func(x))

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x, best_f = x.copy(), f

        # Helper: Nelder-Mead
        def nelder_mead(x0, max_iter=200):
            nonlocal best_x, best_f
            dim = self.dim
            # Initial simplex
            simplex = [x0.copy()]
            f0 = eval_func(x0)
            update_best(x0, f0)
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.5 if xi[i] < 4.5 else -0.5
                simplex.append(xi)

            # Sort simplex
            simplex.sort(key=lambda x: eval_func(x))
            # Actually we need to store function values
            # Let me redo this properly

            # Re-initialize properly
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.5 if xi[i] < 4.5 else -0.5
                simplex.append(xi)

            fvals = [eval_func(x) for x in simplex]
            for x, f in zip(simplex, fvals):
                update_best(x, f)

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

            for _ in range(max_iter):
                if eval_count[0] >= budget or time.time() > deadline:
                    break

                # Sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = eval_func(xr)
                update_best(xr, fr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = eval_func(xe)
                    update_best(xe, fe)
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
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        fc = eval_func(xc)
                        update_best(xc, fc)
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                fvals[i] = eval_func(simplex[i])
                                update_best(simplex[i], fvals[i])
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        fc = eval_func(xc)
                        update_best(xc, fc)
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                fvals[i] = eval_func(simplex[i])
                                update_best(simplex[i], fvals[i])

                if eval_count[0] >= budget or time.time() > deadline:
                    break

                # Check convergence
                if max(fvals) - min(fvals) < 1e-12:
                    break

            return simplex[0], fvals[0]

        # Helper: Estimate Hessian via finite differences
        def estimate_hessian(x, h=0.01):
            dim = self.dim
            H = np.zeros((dim, dim))
            f0 = eval_func(x)
            for i in range(dim):
                xp = x.copy()
                xp[i] += h
                fp = eval_func(xp)
                xm = x.copy()
                xm[i] -= h
                fm = eval_func(xm)
                H[i, i] = (fp - 2 * f0 + fm) / (h * h)
                for j in range(i + 1, dim):
                    xpp = x.copy()
                    xpp[i] += h
                    xpp[j] += h
                    fpp = eval_func(xpp)
                    xpm = x.copy()
                    xpm[i] += h
                    xpm[j] -= h
                    fpm = eval_func(xpm)
                    xmp = x.copy()
                    xmp[i] -= h
                    xmp[j] += h
                    fmp = eval_func(xmp)
                    xmm = x.copy()
                    xmm[i] -= h
                    xmm[j] -= h
                    fxmm = eval_func(xmm)
                    H[i, j] = H[j, i] = (fpp - fpm - fmp + fxmm) / (4 * h * h)
            return H

        # Helper: Newton step
        def newton_step(x):
            dim = self.dim
            # Estimate gradient
            h = 0.01
            grad = np.zeros(dim)
            f0 = eval_func(x)
            for i in range(dim):
                xp = x.copy()
                xp[i] += h
                fp = eval_func(xp)
                grad[i] = (fp - f0) / h

            # Estimate Hessian
            H = estimate_hessian(x, h)

            # Regularize Hessian
            eigvals, eigvecs = np.linalg.eigh(H)
            min_eig = np.min(eigvals)
            if min_eig < 1e-6:
                H += (1e-6 - min_eig) * np.eye(dim)

            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                step = -grad

            # Limit step size
            step_norm = np.linalg.norm(step)
            if step_norm > 5.0:
                step = step * 5.0 / step_norm

            return x + step

        # Main loop: multi-start with NM + Newton polish
        # First, do a few random starts to find a good basin
        n_starts = 3
        start_points = []
        for i in range(n_starts):
            if eval_count[0] >= budget or time.time() > deadline:
                break
            if i == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4, 4, dim)
            start_points.append(x0)

        # For each start, run NM then Newton polish
        for x0 in start_points:
            if eval_count[0] >= budget or time.time() > deadline:
                break

            # Run Nelder-Mead
            x_nm, f_nm = nelder_mead(x0, max_iter=100)
            update_best(x_nm, f_nm)

            if eval_count[0] >= budget or time.time() > deadline:
                break

            # Newton polish: iterate a few times
            x_newton = x_nm.copy()
            for _ in range(3):
                if eval_count[0] >= budget or time.time() > deadline:
                    break
                x_candidate = newton_step(x_newton)
                x_candidate = np.clip(x_candidate, -5, 5)
                f_candidate = eval_func(x_candidate)
                update_best(x_candidate, f_candidate)
                if f_candidate < eval_func(x_newton):
                    x_newton = x_candidate
                else:
                    break

        # If we have budget left, do a final local search around the best point
        if best_x is not None and eval_count[0] < budget and time.time() < deadline:
            # Small (1+1)-ES around best point
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(budget - eval_count[0]):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
