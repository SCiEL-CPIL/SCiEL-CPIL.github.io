"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with periodic re-seeding for multi-modal search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # Count function calls
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            return float(val)

        best_x = None
        best_f = np.inf

        # Phase 1: Multiple Nelder-Mead restarts
        num_restarts = 6
        budget_per_restart = max(20, (budget * 80) // (100 * num_restarts))

        # Starting points
        start_points = [np.zeros(dim)]
        for _ in range(min(4, num_restarts - 1)):
            start_points.append(rng.uniform(-4, 4, dim))
        while len(start_points) < num_restarts:
            start_points.append(rng.uniform(-4.5, 4.5, dim))

        for sp in start_points:
            if time.time() > deadline:
                break
            if call_count >= budget - 10:
                break

            x, fx = self._nelder_mead(
                safe_func, sp, dim, budget_per_restart, deadline, rng,
                initial_step=2.0
            )
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Final high-precision Nelder-Mead from best point
        if best_x is not None and call_count < budget - 5 and time.time() < deadline:
            remaining = budget - call_count - 5
            if remaining > 10:
                x, fx = self._nelder_mead(
                    safe_func, best_x, dim, remaining, deadline, rng,
                    initial_step=0.5
                )
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        if best_x is None:
            # Fallback: should not happen, but just in case
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f

    def _nelder_mead(self, func, x0, dim, max_calls, deadline, rng, initial_step=2.0):
        """Nelder-Mead simplex method with adaptive parameters."""
        # Initialize simplex
        simplex = [x0.copy()]
        simplex_f = [func(x0)]

        # Create other vertices
        for i in range(dim):
            xi = x0.copy()
            if abs(xi[i]) > 1e-10:
                xi[i] *= 1.05
            else:
                xi[i] += 0.05 * initial_step
            xi = np.clip(xi, -5.0, 5.0)
            simplex.append(xi)
            simplex_f.append(func(xi))

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        while time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            # Check convergence
            f_spread = simplex_f[-1] - simplex_f[0]
            if f_spread < 1e-10:
                break

            # Compute centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = func(x_r)

            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = func(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if f_r < simplex_f[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = func(x_c)
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex_f[i] = func(simplex[i])
                else:
                    x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = func(x_c)
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex_f[i] = func(simplex[i])

            # Check if simplex is getting very small
            simplex_spread = np.max([np.linalg.norm(simplex[i] - simplex[0]) for i in range(1, dim + 1)])
            if simplex_spread < 1e-6:
                # Perturb the worst vertex
                worst_idx = int(np.argmax(simplex_f))
                simplex[worst_idx] = simplex[0] + rng.normal(0, 0.5, dim)
                simplex[worst_idx] = np.clip(simplex[worst_idx], -5.0, 5.0)
                simplex_f[worst_idx] = func(simplex[worst_idx])

            # Limit iterations to avoid infinite loops
            if len(simplex_f) > 0:
                break  # This is a placeholder; we need a proper iteration limit

        # Actually, let me fix this - I need a proper loop structure
        # Let me rewrite with a cleaner approach

        return self._nelder_mead_v2(func, x0, dim, max_calls, deadline, rng, initial_step)

    def _nelder_mead_v2(self, func, x0, dim, max_calls, deadline, rng, initial_step=2.0):
        """Clean Nelder-Mead implementation."""
        # Initialize simplex
        simplex = [x0.copy()]
        simplex_f = [func(x0)]

        for i in range(dim):
            xi = x0.copy()
            if abs(xi[i]) > 1e-10:
                xi[i] *= 1.05
            else:
                xi[i] += 0.05 * initial_step
            xi = np.clip(xi, -5.0, 5.0)
            simplex.append(xi)
            simplex_f.append(func(xi))

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        for iteration in range(max_calls):
            if time.time() > deadline:
                break

            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            if simplex_f[-1] - simplex_f[0] < 1e-10:
                break

            centroid = np.mean(simplex[:-1], axis=0)

            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = func(x_r)

            if f_r < simplex_f[0]:
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = func(x_e)
                if f_e < f_r:
                    simplex[-1], simplex_f[-1] = x_e, f_e
                else:
                    simplex[-1], simplex_f[-1] = x_r, f_r
            elif f_r < simplex_f[-2]:
                simplex[-1], simplex_f[-1] = x_r, f_r
            else:
                if f_r < simplex_f[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = func(x_c)
                    if f_c <= f_r:
                        simplex[-1], simplex_f[-1] = x_c, f_c
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex_f[i] = func(simplex[i])
                else:
                    x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = func(x_c)
                    if f_c < simplex_f[-1]:
                        simplex[-1], simplex_f[-1] = x_c, f_c
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex_f[i] = func(simplex[i])

            simplex_spread = np.max([np.linalg.norm(simplex[i] - simplex[0]) for i in range(1, dim + 1)])
            if simplex_spread < 1e-6:
                worst_idx = int(np.argmax(simplex_f))
                simplex[worst_idx] = simplex[0] + rng.normal(0, 0.5, dim)
                simplex[worst_idx] = np.clip(simplex[worst_idx], -5.0, 5.0)
                simplex_f[worst_idx] = func(simplex[worst_idx])

        best_idx = int(np.argmin(simplex_f))
        return simplex[best_idx].copy(), simplex_f[best_idx]
# EVOLVE-BLOCK-END
