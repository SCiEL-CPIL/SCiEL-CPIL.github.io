"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (1+1) variant with success-rate step-size adaptation and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim

        # --- Phase 1: CMA-ES (1+1) global search ---
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx

        sigma = 3.0
        # Success-rate based step size adaptation (CMA-ES inspired)
        # Track successes over a window
        window_size = 10
        successes = 0
        window = 0

        calls_used = 1
        budget_phase1 = int(self.budget * 0.70)

        for i in range(budget_phase1 - 1):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x, fx = y, fy
                successes += 1
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                pass

            window += 1
            if window >= window_size:
                rate = successes / window
                # Target success rate ~ 0.33
                if rate > 0.5:
                    sigma *= 1.3
                elif rate < 0.2:
                    sigma *= 0.7
                else:
                    sigma *= 1.05
                successes = 0
                window = 0

            # Adapt sigma with a floor
            sigma = max(sigma, 1e-6)

        # --- Phase 2: Nelder-Mead polish from best point found ---
        remaining = self.budget - calls_used
        if remaining > 10:
            x0 = best_x.copy()
            f0 = best_fx

            # Build initial simplex
            simplex = [x0.copy()]
            for d in range(dim):
                pt = x0.copy()
                step = 0.1 if abs(x0[d]) > 1e-10 else 0.1
                pt[d] += step
                pt = np.clip(pt, -5, 5)
                simplex.append(pt)

            fvals = [func(pt) for pt in simplex]
            calls_used += dim
            remaining -= dim

            # Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            for _ in range(remaining):
                if time.time() > deadline:
                    break
                if calls_used >= self.budget:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]

                # Reflection
                xr = centroid + alpha * (centroid - worst)
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_used += 1
                if calls_used >= self.budget:
                    break

                if fvals[0] <= fr < fvals[-1]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - worst)
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_used += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + sigma_nm * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - sigma_nm * (xr - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_used += 1
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for j in range(1, dim + 1):
                            simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], -5, 5)
                            fvals[j] = func(simplex[j])
                            calls_used += 1
                            if calls_used >= self.budget:
                                break
                        if calls_used >= self.budget:
                            break

                # Track best
                if fvals[0] < best_fx:
                    best_fx = fvals[0]
                    best_x = simplex[0].copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
