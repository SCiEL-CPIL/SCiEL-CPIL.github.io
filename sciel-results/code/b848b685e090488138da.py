"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with adaptive restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # We'll track best point and best value globally
        global_best_x = None
        global_best_f = np.inf

        # Phase 1: Random initialization to find a good starting region
        # For bent cigar, the function is highly anisotropic
        # We do a brief random search to get a reasonable starting point
        n_init = min(50, budget // 10)
        best_x_init = None
        best_f_init = np.inf
        for i in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f_init:
                best_f_init = fx
                best_x_init = x.copy()
            if time.time() > deadline:
                break

        if best_x_init is None:
            best_x_init = np.zeros(dim)
            best_f_init = func(best_x_init)

        global_best_x = best_x_init.copy()
        global_best_f = best_f_init

        # Phase 2: Nelder-Mead with adaptive step size
        # Initialize simplex
        alpha = 1.0  # expansion
        gamma = 2.0  # reflection
        rho = 0.5    # contraction
        sigma_init = 1.0

        def nm_simplex(x0, f0, max_iter):
            nonlocal global_best_x, global_best_f, budget
            nonlocal deadline, start_time

            # Build initial simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(dim):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += sigma_init

            # Evaluate all simplex points
            fvals = np.zeros(dim + 1)
            fvals[0] = f0
            for i in range(dim):
                fvals[i + 1] = func(simplex[i + 1])
                if fvals[i + 1] < global_best_f:
                    global_best_f = fvals[i + 1]
                    global_best_x = simplex[i + 1].copy()

            for it in range(max_iter):
                if time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Check convergence
                if fvals[-1] - fvals[0] < 1e-10 * max(1.0, abs(fvals[0])):
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + gamma * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                if fr < global_best_f:
                    global_best_f = fr
                    global_best_x = xr.copy()

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + alpha * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    if fe < global_best_f:
                        global_best_f = fe
                        global_best_x = xe.copy()
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
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)
                        if fc < global_best_f:
                            global_best_f = fc
                            global_best_x = xc.copy()
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals[i] = func(simplex[i])
                                if fvals[i] < global_best_f:
                                    global_best_f = fvals[i]
                                    global_best_x = simplex[i].copy()
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)
                        if fc < global_best_f:
                            global_best_f = fc
                            global_best_x = xc.copy()
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals[i] = func(simplex[i])
                                if fvals[i] < global_best_f:
                                    global_best_f = fvals[i]
                                    global_best_x = simplex[i].copy()

            return simplex[0].copy(), fvals[0]

        # Run Nelder-Mead with multiple restarts at different scales
        # The bent cigar has a narrow valley, so we need to be careful
        # with initial step sizes. Use progressively smaller steps.

        # First, try with moderate step size
        remaining_budget = budget - n_init
        if remaining_budget > 10 and time.time() < deadline:
            x_res, f_res = nm_simplex(best_x_init, best_f_init, remaining_budget // 2)
            if f_res < global_best_f:
                global_best_f = f_res
                global_best_x = x_res.copy()

            # Second restart with smaller step if still improving
            if time.time() < deadline and global_best_f > -50:  # not converged yet
                # Try a finer search around the current best
                sigma_fine = 0.1
                x_fine = global_best_x.copy()
                f_fine = global_best_f
                # Quick local NM with small steps
                for _ in range(min(30, budget - n_init)):
                    if time.time() > deadline:
                        break
                    improved = False
                    for d in range(dim):
                        for sign in [1, -1]:
                            xn = x_fine.copy()
                            xn[d] += sign * sigma_fine
                            xn = np.clip(xn, -5, 5)
                            fn = func(xn)
                            if fn < f_fine:
                                x_fine = xn
                                f_fine = fn
                                improved = True
                                if fn < global_best_f:
                                    global_best_f = fn
                                    global_best_x = xn.copy()
                    if not improved:
                        sigma_fine *= 0.5
                        if sigma_fine < 1e-6:
                            break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
