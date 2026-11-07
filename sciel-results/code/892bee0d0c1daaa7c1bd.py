"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s limit

        # Safe call wrapper
        def safe_call(x):
            try:
                val = func(x)
                if not np.isfinite(val):
                    return np.inf
                return float(val)
            except Exception:
                return np.inf

        # Nelder-Mead simplex method
        def nelder_mead(func_safe, x0, budget_remaining, dim, deadline):
            # Initialize simplex
            simplex = []
            fvals = []

            # First vertex
            x0 = np.clip(x0, -5, 5)
            f0 = func_safe(x0)
            simplex.append(x0.copy())
            fvals.append(f0)
            budget_remaining -= 1

            # Other vertices
            for i in range(dim):
                xi = x0.copy()
                # Step size: 0.05 if 0, 0.25 otherwise
                if abs(x0[i]) < 1e-10:
                    xi[i] += 0.05
                else:
                    xi[i] += 0.25 * np.sign(x0[i])
                xi = np.clip(xi, -5, 5)
                fi = func_safe(xi)
                simplex.append(xi)
                fvals.append(fi)
                budget_remaining -= 1
                if time.time() > deadline:
                    break

            if len(simplex) < 2:
                return x0, f0, budget_remaining

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            # Main loop
            while budget_remaining > 0 and time.time() < deadline:
                # Sort by function value
                order = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func_safe(xr)
                budget_remaining -= 1

                if fr < fvals[0]:
                    # Try expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func_safe(xe)
                    budget_remaining -= 1

                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Reflection is good
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
                    fc = func_safe(xc)
                    budget_remaining -= 1

                    if fc <= min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func_safe(simplex[i])
                            budget_remaining -= 1
                            if time.time() > deadline:
                                break
                        if time.time() > deadline:
                            break

                # Check for convergence
                fspread = max(fvals) - min(fvals)
                if fspread < 1e-12:
                    break

            # Return best point
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx], budget_remaining

        # Multi-start Nelder-Mead
        rng = np.random.default_rng(42)

        # Generate diverse starting points
        start_points = []
        # Center
        start_points.append(np.zeros(dim))
        # Corners of [-4, 4]^dim (sampled)
        for _ in range(8):
            pt = rng.uniform(-4, 4, dim)
            start_points.append(pt)
        # Random points
        for _ in range(12):
            pt = rng.uniform(-5, 5, dim)
            start_points.append(pt)

        best_x = None
        best_f = np.inf
        total_budget = budget

        for start in start_points:
            if total_budget <= 0 or time.time() > deadline:
                break

            # Allocate budget to this restart
            # Use remaining budget, but ensure at least dim+1 for simplex
            alloc = min(total_budget, max(dim + 1, 50))
            x, f, remaining = nelder_mead(safe_call, start, alloc, dim, deadline)
            total_budget -= (alloc - remaining)

            if f < best_f:
                best_f = f
                best_x = x

            # If we found a very good solution, do a local refinement
            if best_f < 1e-6 and total_budget > 2 * (dim + 1):
                # Do another NM run from best_x with smaller initial step
                x, f, remaining = nelder_mead(safe_call, best_x, min(total_budget, 30), dim, deadline)
                total_budget -= (30 - remaining)
                if f < best_f:
                    best_f = f
                    best_x = x

        if best_x is None:
            # Fallback: evaluate origin
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
