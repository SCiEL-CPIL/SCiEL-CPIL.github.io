"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Nelder-Mead with multi-start and coordinate descent polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, fx0, budget, deadline, rng):
        """Run Nelder-Mead simplex optimization. Returns (best_x, best_fx, calls_used)."""
        dim = self.dim
        calls = 1  # fx0 already evaluated

        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [fx0]
        alpha = 0.5  # initial step size for simplex expansion

        for i in range(dim):
            xi = x0.copy()
            xi[i] += alpha
            simplex.append(xi)
            fi = func(xi)
            fvals.append(fi)
            calls += 1

        best_idx = int(np.argmin(fvals))
        best_x = simplex[best_idx].copy()
        best_fx = fvals[best_idx]

        # Nelder-Mead parameters
        alpha_n = 1.0   # reflection
        gamma = 2.0     # expansion
        rho = 0.5       # contraction
        sigma_s = 0.5   # shrink

        max_iter = budget - calls
        stall_count = 0
        stall_limit = 20

        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Check for convergence
            if fvals[-1] - fvals[0] < 1e-12:
                break

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha_n * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
                stall_count = 0
            elif fr < fvals[-2]:
                # Accept reflection
                simplex[-1] = xr
                fvals[-1] = fr
                stall_count = 0
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
                calls += 1

                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                    stall_count = 0
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_s * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget:
                            break
                    stall_count += 1

            # Update best
            new_best_idx = int(np.argmin(fvals))
            if fvals[new_best_idx] < best_fx:
                best_x = simplex[new_best_idx].copy()
                best_fx = fvals[new_best_idx]

            # Stall detection
            if stall_count >= stall_limit:
                # Random restart within current region
                x_new = best_x + 1.0 * rng.standard_normal(dim)
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls += 1
                if f_new < best_fx:
                    best_x = x_new
                    best_fx = f_new
                # Reinitialize simplex around new best
                simplex = [best_x.copy()]
                fvals = [best_fx]
                for i in range(dim):
                    xi = best_x.copy()
                    xi[i] += 0.5
                    simplex.append(xi)
                    fi = func(xi)
                    fvals.append(fi)
                    calls += 1
                stall_count = 0

        return best_x, best_fx, calls

    def _coordinate_descent(self, func, x0, fx0, budget, deadline):
        """Coarse-to-fine coordinate descent. Returns (best_x, best_fx, calls_used)."""
        dim = self.dim
        calls = 1
        x = x0.copy()
        fx = fx0
        best_x = x.copy()
        best_fx = fx

        step_sizes = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]

        for step in step_sizes:
            if time.time() > deadline or calls >= budget:
                break

            improved = True
            while improved and calls < budget and time.time() <= deadline:
                improved = False
                for i in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break

                    # Try positive step
                    xp = x.copy()
                    xp[i] += step
                    xp = np.clip(xp, -5, 5)
                    fp = func(xp)
                    calls += 1
                    if fp < fx:
                        x = xp
                        fx = fp
                        improved = True
                        if fx < best_fx:
                            best_x = x.copy()
                            best_fx = fx

                    # Try negative step
                    if calls >= budget or time.time() > deadline:
                        break
                    xm = x.copy()
                    xm[i] -= step
                    xm = np.clip(xm, -5, 5)
                    fm = func(xm)
                    calls += 1
                    if fm < fx:
                        x = xm
                        fx = fm
                        improved = True
                        if fx < best_fx:
                            best_x = x.copy()
                            best_fx = fx

        return best_x, best_fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s kill
        rng = np.random.default_rng(42)

        total_budget = self.budget
        dim = self.dim

        best_x = None
        best_fx = np.inf
        total_calls = 0

        # Phase 1: Random sampling to find promising regions
        n_samples = 20
        for i in range(n_samples):
            if total_calls >= total_budget or time.time() > deadline:
                break
            x0 = rng.uniform(-5, 5, dim)
            fx0 = func(x0)
            total_calls += 1
            if fx0 < best_fx:
                best_x = x0.copy()
                best_fx = fx0

        if best_x is None:
            x0 = rng.uniform(-5, 5, dim)
            fx0 = func(x0)
            best_x = x0
            best_fx = fx0
            total_calls += 1

        # Phase 2: Nelder-Mead from best point
        remaining = total_budget - total_calls
        if remaining > 10 and time.time() < deadline:
            nm_budget = min(remaining, 200)
            bx, bfx, calls_used = self._nelder_mead(
                func, best_x, best_fx, nm_budget + 1, deadline, rng
            )
            total_calls += calls_used - 1
            if bfx < best_fx:
                best_x = bx
                best_fx = bfx

        # Phase 3: Coordinate descent polish
        remaining = total_budget - total_calls
        if remaining > 5 and time.time() < deadline:
            cd_budget = min(remaining, 80)
            bx, bfx, calls_used = self._coordinate_descent(
                func, best_x, best_fx, cd_budget + 1, deadline
            )
            total_calls += calls_used - 1
            if bfx < best_fx:
                best_x = bx
                best_fx = bfx

        # Phase 4: If budget remains, do another NM round
        remaining = total_budget - total_calls
        if remaining > 10 and time.time() < deadline:
            nm_budget = min(remaining, 100)
            bx, bfx, calls_used = self._nelder_mead(
                func, best_x, best_fx, nm_budget + 1, deadline, rng
            )
            total_calls += calls_used - 1
            if bfx < best_fx:
                best_x = bx
                best_fx = bfx

        return best_x, best_fx
# EVOLVE-BLOCK-END
