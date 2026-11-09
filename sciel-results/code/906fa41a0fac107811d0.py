"""Rewritten optimizer: Structure-exploiting grid search + Nelder-Mead for Rastrigin f3."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic grid search + Nelder-Mead for Rastrigin (f3) on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # Safety margin
        d = self.dim
        budget = self.budget
        best_x = None
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal best_x, best_f, calls
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse-to-fine 1D grid search to find the integer lattice point
        # Rastrigin's global minimum is at the origin in its native frame.
        # After hidden shift/rotation, it's at some point in [-4, 4]^5.
        # We perform a coarse grid search first, then refine.

        # Coarse grid: step 1.0, so we check integer-like points
        # Actually, let's do a multi-resolution grid search.
        # Start with a very coarse grid to find the basin.
        
        # For each dimension, we'll search independently first (separable assumption for coarse phase)
        # But Rastrigin is separable: f(x) = sum(10*d^2 + x_i^2*(1-cos(2*pi*x_i)) - 10*d)
        # Wait, standard Rastrigin: f(x) = 10*A + sum(x_i^2 - A*cos(2*pi*x_i)), A=10
        # It's separable! So we can optimize each dimension independently if the shift is axis-aligned.
        # But the task says "hidden seeded shift, rotation and offset". Rotation breaks separability.
        # So we can't just do 1D grid search per axis if there's rotation.
        # However, the inspiration says "Exploit Rastrigin's separable periodic structure by performing a coarse-to-fine 1D grid search along each coordinate axis to identify the global basin's integer lattice point".
        # This assumes the rotation is small or the grid is fine enough to catch the basin.
        # Let's try a coarse grid in 5D first, but that's 10^5 points which is too many for budget 500.
        # So we must be clever.
        
        # Alternative: Use a few random starts + Nelder-Mead.
        # The previous best score (1.479) used (1+1)-ES with 1/5 rule + Nelder-Mead.
        # The directive asks for a STRUCTURAL change: "Exploit Rastrigin's separable periodic structure by performing a coarse-to-fine 1D grid search along each coordinate axis".
        # Even with rotation, if we do a grid search with step ~1, we might find a point close to the basin.
        # Let's do a coordinate-wise grid search: for each dimension, sweep that dimension while keeping others fixed at 0 (or current best).
        # This is a form of coordinate descent with grid steps.

        # Phase 1: Coordinate-wise coarse grid search
        # Start at 0
        x = np.zeros(d)
        fx = safe_call(x)
        if fx is None:
            return best_x, best_f

        # For each dimension, sweep from -5 to 5 with step 1
        for dim_idx in range(d):
            if time.time() > deadline or calls >= budget:
                break
            for val in np.arange(-5, 5.1, 1.0):
                if time.time() > deadline or calls >= budget:
                    break
                x_new = x.copy()
                x_new[dim_idx] = val
                fx_new = safe_call(x_new)
                if fx_new is not None and fx_new < fx:
                    fx = fx_new
                    x = x_new

        # Phase 2: Refine with smaller steps
        for step in [0.5, 0.25, 0.1, 0.05]:
            if time.time() > deadline or calls >= budget:
                break
            for dim_idx in range(d):
                if time.time() > deadline or calls >= budget:
                    break
                # Local search around current x[dim_idx]
                center = x[dim_idx]
                # Search a window of +/- 1*step around center
                for offset in np.arange(-1, 1.1, 1.0):
                    if time.time() > deadline or calls >= budget:
                        break
                    x_new = x.copy()
                    x_new[dim_idx] = center + offset * step
                    fx_new = safe_call(x_new)
                    if fx_new is not None and fx_new < fx:
                        fx = fx_new
                        x = x_new
                        center = x_new[dim_idx]

        # Phase 3: Nelder-Mead polish
        if time.time() < deadline and calls < budget - 10:
            # Implement a simple Nelder-Mead
            nm_x = best_x.copy()
            nm_f = best_f
            # Initial simplex
            simplex = [nm_x.copy()]
            for i in range(d):
                pt = nm_x.copy()
                pt[i] += 0.1 if pt[i] < 4.9 else -0.1
                simplex.append(pt)
            fvals = [nm_f]
            for pt in simplex[1:]:
                if time.time() > deadline or calls >= budget:
                    break
                fp = safe_call(pt)
                if fp is None:
                    break
                fvals.append(fp)

            # Nelder-Mead iterations
            max_iter = 100
            for _ in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break
                # Sort
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                # Check convergence
                if abs(fvals[0] - fvals[-1]) < 1e-8:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = fvals[-1]

                # Reflection
                xr = centroid + 1.0 * (centroid - worst)
                xr = np.clip(xr, -5, 5)
                fxr = safe_call(xr)
                if fxr is None:
                    break

                if fxr < fvals[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - worst)
                    xe = np.clip(xe, -5, 5)
                    fxe = safe_call(xe)
                    if fxe is None:
                        break
                    if fxe < fxr:
                        simplex[-1] = xe
                        fvals[-1] = fxe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fxr
                elif fxr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fxr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (worst - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = safe_call(xc)
                    if fxc is None:
                        break
                    if fxc < worst_f:
                        simplex[-1] = xc
                        fvals[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            if time.time() > deadline or calls >= budget:
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fxi = safe_call(simplex[i])
                            if fxi is None:
                                break
                            fvals[i] = fxi

            if best_x is not None and best_f < nm_f:
                pass # best_x is already updated by safe_call

        return best_x, best_f
# EVOLVE-BLOCK-END
