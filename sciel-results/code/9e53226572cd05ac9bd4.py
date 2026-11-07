"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with global-to-local strategy for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Start with a reasonably informed initial point
        # For Schwefel-like functions, the global minimum is often near specific values
        # but we don't know the shift, so we start centered but with some spread
        x0 = np.zeros(dim)
        fx0 = func(x0)

        best_x = x0.copy()
        best_f = fx0

        calls_made = 1

        # Nelder-Mead simplex method
        # Create initial simplex
        def nm_simplex(func, x_start, f_start, max_iter, initial_scale, deadline):
            nonlocal calls_made, best_x, best_f

            n = len(x_start)
            # Create simplex: n+1 points
            simplex = np.zeros((n + 1, n))
            simplex[0] = x_start
            fvals = np.zeros(n + 1)
            fvals[0] = f_start

            for i in range(1, n + 1):
                simplex[i] = x_start.copy()
                simplex[i][i - 1] += initial_scale
                simplex[i] = np.clip(simplex[i], -5, 5)
                fvals[i] = func(simplex[i])
                calls_made += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = simplex[i].copy()

            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                if calls_made >= budget - 5:  # reserve some calls
                    break

                # Sort simplex by function value
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-12:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_made += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_made += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
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
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_made += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()

                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls_made += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()

                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

            return best_x, best_f, calls_made

        # Strategy: Multiple restarts with decreasing simplex scale
        # Phase 1: Large-scale exploration
        # Phase 2: Medium-scale refinement
        # Phase 3: Fine-scale convergence

        # Use multiple random starting points for global search
        n_restarts = min(10, max(3, budget // 50))
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            if calls_made >= budget - 10:
                break

            # Generate a starting point
            if restart == 0:
                x_start = x0.copy()
                f_start = fx0
            else:
                # Random start with some bias toward center
                x_start = rng.uniform(-4, 4, dim)
                f_start = func(x_start)
                calls_made += 1
                if f_start < best_f:
                    best_f = f_start
                    best_x = x_start.copy()

            # Determine simplex scale based on phase
            if restart < n_restarts // 3:
                scale = 3.0  # Large scale for global exploration
            elif restart < 2 * n_restarts // 3:
                scale = 1.0  # Medium scale
            else:
                scale = 0.3  # Small scale for local refinement

            # Allocate iterations based on remaining budget
            remaining = budget - calls_made
            max_iter = max(10, min(remaining // (n_restarts - restart), 100))

            _, _, calls_made = nm_simplex(
                func, x_start, f_start, max_iter, scale, deadline
            )

            if calls_made >= budget - 5:
                break

        # Final polish: Very fine Nelder-Mead around best point
        if time.time() < deadline and calls_made < budget - 5:
            remaining = budget - calls_made
            if remaining > 10:
                max_iter = min(remaining - 5, 50)
                _, _, calls_made = nm_simplex(
                    func, best_x, best_f, max_iter, 0.05, deadline
                )

        return best_x, best_f
# EVOLVE-BLOCK-END
