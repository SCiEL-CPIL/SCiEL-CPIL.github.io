"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Grid-based global search followed by Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return None
            calls_made += 1
            try:
                val = func(x)
                return float(val)
            except Exception:
                return None

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def nelder_mead(x0, max_iter, initial_step=1.0):
            """Nelder-Mead simplex optimization."""
            nonlocal best_x, best_f, calls_made, deadline

            dim = self.dim
            # Build initial simplex
            simplex = [x0.copy()]
            for i in range(dim):
                pert = x0.copy()
                # Use a slightly larger step for initial exploration if in a flat region,
                # but keep it within bounds
                step = initial_step
                pert[i] = pert[i] + step
                simplex.append(np.clip(pert, -5, 5))

            # Evaluate initial simplex
            fvals = []
            for i in range(dim + 1):
                if time.time() > deadline or calls_made >= self.budget:
                    return
                f = safe_call(simplex[i])
                if f is None:
                    f = np.inf
                fvals.append(f)
                update_best(simplex[i], f)

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
                        update_best(xe, fe)
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                        update_best(xr, fr)
                elif fr < fvals[-2]:
                    # Simple reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                    update_best(xr, fr)
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
                        update_best(xc, fc)
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f = safe_call(simplex[i])
                            if f is None:
                                f = np.inf
                            fvals[i] = f
                            update_best(simplex[i], f)

                # Check convergence
                if len(fvals) == dim + 1:
                    f_range = max(fvals) - min(fvals)
                    if f_range < 1e-12:
                        break

            return

        # Phase 1: Coarse Grid Search
        # The minimum is in [-4, 4]^5.
        # We use a 3-point grid per dimension: -4, 0, 4.
        # Total points: 3^5 = 243. This fits within the 500 budget.
        if self.dim == 5:
            grid_coords = [-4.0, 0.0, 4.0]
            # Generate all combinations
            import itertools
            grid_points = list(itertools.product(grid_coords, repeat=self.dim))
        else:
            # Fallback for other dimensions if needed, though task specifies dim=5
            grid_coords = np.linspace(-4.0, 4.0, 3)
            grid_points = []
            for i in range(3 ** self.dim):
                p = []
                for d in range(self.dim):
                    p.append(grid_coords[(i // (3 ** (self.dim - 1 - d))) % 3])
                grid_points.append(p)

        # Evaluate grid points
        for p in grid_points:
            if time.time() > deadline or calls_made >= self.budget:
                break
            x = np.array(p, dtype=float)
            f = safe_call(x)
            if f is not None:
                update_best(x, f)

        # Phase 2: Local Refinement with Nelder-Mead
        if best_x is not None and calls_made < self.budget:
            remaining = self.budget - calls_made
            # Allocate most of the remaining budget to Nelder-Mead
            max_iter = min(remaining - 2, 200)
            # Initial step size: Since we are in a coarse grid, the valley might be nearby.
            # A step of 1.0 is reasonable.
            nelder_mead(best_x, max_iter, initial_step=1.0)

        # Phase 3: If budget remains, try a few random restarts or finer local search
        if calls_made < self.budget and time.time() < deadline:
            # Try a few random restarts to see if we missed a better basin
            for attempt in range(5):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                x = rng.uniform(-4, 4, self.dim)
                f = safe_call(x)
                if f is None:
                    continue
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    # If we found a new best, run a short NM
                    if calls_made < self.budget - 10:
                        remaining = self.budget - calls_made
                        max_iter = min(remaining - 2, 50)
                        nelder_mead(best_x, max_iter, initial_step=0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
