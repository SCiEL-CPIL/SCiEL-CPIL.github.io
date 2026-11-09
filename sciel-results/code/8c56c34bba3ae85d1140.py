"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive step sizes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track global best
        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= budget:
                return np.inf
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            calls_used += 1
            try:
                val = func(x_clipped)
                return float(val)
            except Exception:
                return np.inf

        # Phase 1: Diversified sampling to find good basins
        # Use a mix of random points and structured patterns
        num_initial = min(50, budget // 10)
        initial_points = []

        # Center
        initial_points.append(np.zeros(dim))

        # Axis-aligned points
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 3.0
                initial_points.append(p)

        # Random points
        while len(initial_points) < num_initial:
            p = rng.uniform(-4, 4, dim)
            initial_points.append(p)

        # Evaluate initial points
        local_bests = []
        for p in initial_points:
            if calls_used >= budget or time.time() - start_time > time_limit:
                break
            f = safe_call(p)
            if f < best_f:
                best_f = f
                best_x = p.copy()
            local_bests.append((f, p.copy()))

        # Sort by fitness
        local_bests.sort(key=lambda x: x[0])

        # Phase 2: Nelder-Mead from top diverse starting points
        # Use Nelder-Mead on the best few starting points
        num_restarts = min(5, len(local_bests))
        budget_per_restart = max(20, (budget - calls_used) // num_restarts)

        for idx in range(num_restarts):
            if calls_used >= budget or time.time() - start_time > time_limit:
                break

            f0, x0 = local_bests[idx]
            # Skip if this is too similar to a previous restart
            if idx > 0:
                # Check distance to ensure diversity
                prev_best_x = best_x if best_x is not None else np.zeros(dim)
                if np.linalg.norm(x0 - prev_best_x) < 0.5:
                    continue

            # Nelder-Mead implementation
            x_nm = x0.copy()
            f_nm = f0

            # Initial simplex
            simplex = [x_nm.copy()]
            f_simplex = [f_nm]

            for i in range(dim):
                xi = x_nm.copy()
                # Choose step size based on function scale
                step = 0.5
                xi[i] += step
                fi = safe_call(xi)
                if fi == np.inf:
                    fi = f_nm + 1.0
                simplex.append(xi)
                f_simplex.append(fi)

            # Nelder-Mead iterations
            max_nm_iter = budget_per_restart - (dim + 1)
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

            for it in range(max_nm_iter):
                if calls_used >= budget or time.time() - start_time > time_limit:
                    break

                # Sort simplex
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]

                # Update global best
                if f_simplex[0] < best_f:
                    best_f = f_simplex[0]
                    best_x = simplex[0].copy()

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = safe_call(xr)

                if fr < f_simplex[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = safe_call(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                elif fr < f_simplex[-2]:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = safe_call(xc)
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_simplex[i] = safe_call(simplex[i])

                # Check convergence
                if len(f_simplex) > 1:
                    spread = max(f_simplex) - min(f_simplex)
                    if spread < 1e-10:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
