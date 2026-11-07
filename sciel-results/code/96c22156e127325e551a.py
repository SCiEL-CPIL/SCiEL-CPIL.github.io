"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Nelder-Mead with multi-start and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Multi-start sampling to find a good initial point
        # Use 20% of budget for random sampling
        n_samples = max(10, budget // 5)
        best_x = None
        best_f = np.inf

        for i in range(n_samples):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            best_f = func(best_x)

        # Phase 2: Nelder-Mead local optimization from best start
        # Use remaining budget for Nelder-Mead
        remaining_budget = budget - n_samples
        if remaining_budget > 0 and time.time() < deadline:
            nm_x, nm_f = self._nelder_mead(
                func, best_x, remaining_budget, dim, deadline
            )
            if nm_f < best_f:
                best_x, best_f = nm_x, nm_f

        # Phase 3: If time remains, try a few more random restarts with short NM
        if time.time() < deadline - 30:
            n_extra = max(3, (budget - n_samples) // 4)
            for i in range(n_extra):
                if time.time() > deadline - 10:
                    break
                x = rng.uniform(-5, 5, dim)
                f = func(x)
                if f < best_f:
                    best_x, best_f = x.copy(), f
                    # Short Nelder-Mead from this point
                    if time.time() < deadline - 10:
                        nm_x, nm_f = self._nelder_mead(
                            func, best_x, min(30, remaining_budget // 2), dim, deadline
                        )
                        if nm_f < best_f:
                            best_x, best_f = nm_x, nm_f

        return best_x, best_f

    def _nelder_mead(self, func, x0, budget, dim, deadline):
        """Nelder-Mead simplex method."""
        # Initialize simplex
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            if xi[i] == 0:
                xi[i] += 0.25
            else:
                xi[i] += 0.25 * abs(xi[i])
            xi[i] = np.clip(xi[i], -5, 5)
            simplex.append(xi)

        fvals = [func(s) for s in simplex]
        best_f = min(fvals)
        best_x = simplex[fvals.index(best_f)].copy()

        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        calls_used = dim + 1
        max_iter = budget

        for it in range(max_iter):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Check convergence
            if abs(fvals[-1] - fvals[0]) < 1e-10:
                break

            # Centroid of all points except worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_used += 1

            if fr < fvals[0]:
                # Try expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls_used += 1
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
                    xc = centroid + rho * (simplex[-1] - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls_used += 1

                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_used += 1

            # Track best
            current_best = min(fvals)
            if current_best < best_f:
                best_f = current_best
                best_x = simplex[fvals.index(current_best)].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
