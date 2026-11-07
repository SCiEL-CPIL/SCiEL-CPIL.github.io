"""Optimizer for BBOB f9 (Rosenbrock rotated) using multi-start gradient descent with adaptive refinement."""
import numpy as np
import time


class Optimizer:
    """Multi-start gradient-based optimizer with adaptive step-size and Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        calls_made = 0

        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def estimate_gradient(x, eps=1e-5):
            """Central finite difference gradient."""
            grad = np.zeros(dim)
            for i in range(dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] += eps
                xm[i] -= eps
                xp = np.clip(xp, -5, 5)
                xm = np.clip(xm, -5, 5)
                fp = safe_func(xp)
                fm = safe_func(xm)
                grad[i] = (fp - fm) / (2 * eps)
            return grad

        def gradient_descent(x0, fx0, max_iters, eps=1e-5, lr_init=0.05):
            """Gradient descent with backtracking line search."""
            nonlocal budget, calls_made
            x = x0.copy()
            fx = fx0
            lr = lr_init
            iters = 0

            for _ in range(max_iters):
                if time.time() > deadline - 5:
                    break
                if calls_made >= budget:
                    break

                # Estimate gradient (2*dim function calls)
                if calls_made + 2 * dim > budget:
                    break

                grad = estimate_gradient(x, eps)

                # Check if we're at a minimum
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-8:
                    break

                # Backtracking line search
                step_dir = -grad / gnorm  # normalize direction
                alpha = lr_init
                improved = False

                for _ in range(25):
                    if calls_made >= budget:
                        break
                    x_new = np.clip(x + alpha * step_dir, -5, 5)
                    fx_new = safe_func(x_new)

                    if fx_new < fx:
                        x, fx = x_new, fx_new
                        improved = True
                        # Increase learning rate slightly if we made good progress
                        alpha *= 1.15
                        break
                    else:
                        alpha *= 0.5

                if not improved:
                    if alpha < 1e-10:
                        break

                iters += 1

            return x, fx

        def nelder_mead(x0, max_iters=50, tol=1e-6):
            """Nelder-Mead simplex method for local refinement."""
            nonlocal calls_made
            x = x0.copy()
            fx = safe_func(x)

            # Initialize simplex
            simplex = [x.copy()]
            fvals = [fx]
            for i in range(dim):
                xi = x.copy()
                xi[i] += 0.01
                simplex.append(xi)
                fvals.append(safe_func(xi))

            for _ in range(max_iters):
                if time.time() > deadline - 5:
                    break
                if calls_made >= budget:
                    break
                if len(simplex) > dim + 1:
                    break

                # Sort simplex by fitness
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                if abs(fvals[-1] - fvals[0]) < tol:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + (centroid - simplex[-1])
                f_ref = safe_func(x_ref)

                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + 2 * (centroid - simplex[-1])
                    f_exp = safe_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + 0.5 * (simplex[-1] - centroid)
                    f_con = safe_func(x_con)
                    if f_con < fvals[-1]:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = safe_func(simplex[i])

            # Return best point
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # Phase 1: Coarse random search to find promising regions
        n_coarse = min(15, budget // 10)
        coarse_points = []
        for i in range(n_coarse):
            if time.time() > deadline - 10:
                break
            if calls_made >= budget:
                break

            if i == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4, 4, dim)

            fx0 = safe_func(x0)
            coarse_points.append((fx0, x0.copy()))

        # Sort by fitness and take top candidates
        coarse_points.sort(key=lambda t: t[0])
        n_starts = min(5, len(coarse_points))

        # Phase 2: Gradient descent from best starting points
        for s in range(n_starts):
            if time.time() > deadline - 10:
                break
            if calls_made >= budget:
                break

            fx0, x0 = coarse_points[s]

            # Run gradient descent with adaptive epsilon
            eps_start = 1e-4
            eps_end = 1e-6
            lr_start = 0.1
            lr_end = 0.01

            # First pass with larger steps
            if calls_made + 2 * dim < budget:
                x, fx = gradient_descent(x0, fx0, max_iters=50, eps=eps_start, lr_init=lr_start)

            # Second pass with smaller steps for refinement
            if calls_made + 2 * dim < budget and time.time() < deadline - 15:
                x, fx = gradient_descent(x, fx, max_iters=50, eps=eps_end, lr_init=lr_end)

        # Phase 3: Nelder-Mead refinement on the best solution
        if best_x is not None and calls_made + 10 < budget and time.time() < deadline - 15:
            x_nm, fx_nm = nelder_mead(best_x, max_iters=40)

        # Phase 4: Final fine-tuning with very small gradient steps
        if best_x is not None and calls_made + 2 * dim < budget and time.time() < deadline - 10:
            x_final, fx_final = gradient_descent(best_x, best_f, max_iters=30, eps=1e-7, lr_init=0.005)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        return best_x, best_f
