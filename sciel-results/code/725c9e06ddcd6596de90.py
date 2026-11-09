"""Optimizer for BBOB f7 (Step ellipsoidal) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive coordinate + Nelder-Mead hybrid with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280.0  # safety margin before 300s kill
        calls = [0]

        def f(x):
            calls[0] += 1
            return float(func(x))

        rng = np.random.default_rng(7)
        dim = self.dim
        budget = self.budget

        # --- Phase 1: Adaptive coordinate search (fast early progress) ---
        # Start from center; the minimizer is known to be in [-4,4]^5
        x = np.zeros(dim)
        fx = f(x)
        best_x = x.copy()
        best_f = fx

        # Adaptive step sizes per dimension
        steps = np.full(dim, 2.0)

        # Coordinate search with adaptive steps
        coord_budget = min(budget // 2, 200)
        for _ in range(coord_budget):
            if calls[0] >= budget or time.time() > deadline:
                break
            improved = False
            for d in range(dim):
                if calls[0] >= budget or time.time() > deadline:
                    break
                for sign in (1.0, -1.0):
                    if calls[0] >= budget:
                        break
                    trial = best_x.copy()
                    trial[d] += sign * steps[d]
                    trial = np.clip(trial, -5.0, 5.0)
                    ft = f(trial)
                    if ft < best_f:
                        best_f = ft
                        best_x = trial
                        steps[d] *= 1.3
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                # Shrink all steps
                steps *= 0.7
                if np.max(steps) < 1e-4:
                    break

        # --- Phase 2: Nelder-Mead local refinement ---
        if calls[0] < budget and time.time() < deadline:
            # Build simplex around best_x
            simplex = [best_x.copy()]
            for d in range(dim):
                pt = best_x.copy()
                step = max(steps[d] * 0.5, 0.05)
                pt[d] += step
                pt = np.clip(pt, -5.0, 5.0)
                simplex.append(pt)

            # Nelder-Mead implementation
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            n = len(simplex)
            fvals = np.array([f(s) for s in simplex])

            for _ in range(budget - calls[0]):
                if calls[0] >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = fvals[order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = f(xr)
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = f(xe)
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
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = f(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = f(simplex[i])

                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
