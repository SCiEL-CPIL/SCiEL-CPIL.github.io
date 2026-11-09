"""Rewritten optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Two-phase optimizer:
      Phase 1: Adaptive random search with momentum (fast, cheap per step)
      Phase 2: Nelder-Mead local refinement (exploits local structure)
    
    Includes explicit wall-clock deadline guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # --- Phase 1: Adaptive random search with momentum ---
        # Start from center (minimizer is in [-4,4]^5, center 0 is a good prior)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        sigma = 3.0  # initial step size
        momentum = np.zeros(dim)
        momentum_beta = 0.5
        success_count = 0
        window = 10  # adapt sigma every 10 steps

        # Phase 1 budget: use 60% of total budget
        phase1_budget = int(budget * 0.6)

        for i in range(phase1_budget):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Generate candidate
            step = sigma * rng.standard_normal(dim)
            momentum = momentum_beta * momentum + (1 - momentum_beta) * step
            y = np.clip(x + momentum, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                success_count += 1
                x, fx = y, fy
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            else:
                success_count -= 1

            # Adapt sigma every `window` steps
            if (i + 1) % window == 0:
                if success_count > window * 0.5:
                    sigma *= 1.3
                elif success_count < window * 0.3:
                    sigma *= 0.7
                else:
                    sigma *= 0.95
                sigma = np.clip(sigma, 0.01, 10.0)
                success_count = 0

        # --- Phase 2: Nelder-Mead refinement from best point ---
        # Use remaining budget for NM
        remaining = budget - calls
        if remaining > dim + 1 and time.time() < deadline:
            # Initialize simplex around best_x
            simplex = [best_x.copy()]
            for d in range(dim):
                perturbation = best_x.copy()
                perturbation[d] += 0.1
                simplex.append(np.clip(perturbation, -5, 5))

            # Evaluate initial simplex
            simplex_f = [func(s) for s in simplex]
            calls += dim

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            max_nm_iter = min(remaining, 500)

            for _ in range(max_nm_iter):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break

                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Compute centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                fr = func(xr)
                calls += 1

                if simplex_f[0] <= fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                    continue

                if fr < simplex_f[0]:
                    # Expansion
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                    continue

                # Contraction
                if fr < simplex_f[-1]:
                    xc = np.clip(centroid + rho * (centroid - simplex[-1]), -5, 5)
                else:
                    xc = np.clip(centroid - rho * (simplex[-1] - centroid), -5, 5)
                fc = func(xc)
                calls += 1

                if fc < min(fr, simplex_f[-1]):
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    # Shrink
                    for j in range(1, len(simplex)):
                        simplex[j] = np.clip(simplex[0] + sigma_nm * (simplex[j] - simplex[0]), -5, 5)
                        simplex_f[j] = func(simplex[j])
                        calls += 1
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break

                # Update best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f:
                    best_f = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
