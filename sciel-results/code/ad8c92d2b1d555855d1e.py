"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart Nelder-Mead with random initialization for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return None
            calls_made += 1
            try:
                return func(x)
            except Exception:
                return None

        def nelder_mead(x0, max_iter, initial_step=1.0):
            """Nelder-Mead simplex optimization."""
            nonlocal best_x, best_f, calls_made, deadline

            dim = self.dim
            # Build initial simplex
            simplex = [x0.copy()]
            for i in range(dim):
                pert = x0.copy()
                pert[i] = pert[i] + initial_step
                simplex.append(np.clip(pert, -5, 5))

            # Evaluate initial simplex
            fvals = []
            for i in range(dim + 1):
                if time.time() > deadline or calls_made >= self.budget:
                    break
                f = safe_call(simplex[i])
                if f is None:
                    f = np.inf
                fvals.append(f)
                if f < best_f:
                    best_f = f
                    best_x = simplex[i].copy()

            if len(fvals) < dim + 1:
                return

            alpha, gamma, rho = 1.0, 2.0, 0.5  # reflection, expansion, contraction
            shrink = 0.5

            for _ in range(max_iter):
                if time.time() > deadline or calls_made >= self.budget:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = safe_call(xr)
                if fr is None:
                    fr = np.inf

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    fe = safe_call(xe)
                    if fe is None:
                        fe = np.inf
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                        if fe < best_f:
                            best_f = fe
                            best_x = xe.copy()
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                        if fr < best_f:
                            best_f = fr
                            best_x = xr.copy()
                elif fr < fvals[-2]:
                    # Simple reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                    if fr < best_f:
                        best_f = fr
                        best_x = xr.copy()
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = safe_call(xc)
                    if fc is None:
                        fc = np.inf

                    if fc <= min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                        if fc < best_f:
                            best_f = fc
                            best_x = xc.copy()
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f = safe_call(simplex[i])
                            if f is None:
                                f = np.inf
                            fvals[i] = f
                            if f < best_f:
                                best_f = f
                                best_x = simplex[i].copy()

                # Check convergence
                if len(fvals) == dim + 1:
                    f_range = max(fvals) - min(fvals)
                    if f_range < 1e-12:
                        break

            return

        # Multi-restart strategy
        # First, do a coarse random search to find a good starting point
        n_coarse = min(50, self.budget // 10)
        for i in range(n_coarse):
            if time.time() > deadline or calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = safe_call(x)
            if f is not None and f < best_f:
                best_f = f
                best_x = x.copy()

        # Then do Nelder-Mead from the best point found
        if best_x is not None and calls_made < self.budget:
            remaining = self.budget - calls_made
            max_iter = min(remaining - 2, 200)
            nelder_mead(best_x, max_iter, initial_step=1.0)

        # If we have budget left, do another restart from a different region
        if calls_made < self.budget and time.time() < deadline:
            # Try a few more random starts with Nelder-Mead
            for attempt in range(3):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, self.dim)
                f = safe_call(x)
                if f is None:
                    continue
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

                if calls_made >= self.budget or time.time() > deadline:
                    break

                remaining = self.budget - calls_made
                if remaining < 10:
                    break
                max_iter = min(remaining - 2, 100)
                nelder_mead(x, max_iter, initial_step=0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
