"""Nelder-Mead based optimizer for BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex method with adaptive step sizing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initialize simplex: one point at center, others offset along axes
        x0 = np.zeros(dim)
        simplex = [x0.copy()]
        fx_list = [func(x0)]

        # Add vertices with small offsets
        for i in range(dim):
            v = x0.copy()
            v[i] = 1.0
            simplex.append(v)
            fx_list.append(func(v))

        calls = dim + 1
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        # Track best
        best_idx = int(np.argmin(fx_list))
        best_x = simplex[best_idx].copy()
        best_f = fx_list[best_idx]

        while calls < budget:
            # Sort simplex by function value
            order = np.argsort(fx_list)
            simplex = [simplex[i].copy() for i in order]
            fx_list = [fx_list[i] for i in order]

            # Compute centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(np.clip(xr, -5, 5))
            calls += 1

            if fr < fx_list[0]:
                # Try expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = func(np.clip(xe, -5, 5))
                calls += 1
                if fe < fr:
                    simplex[-1] = np.clip(xe, -5, 5)
                    fx_list[-1] = fe
                else:
                    simplex[-1] = np.clip(xr, -5, 5)
                    fx_list[-1] = fr
            elif fr < fx_list[-2]:
                simplex[-1] = np.clip(xr, -5, 5)
                fx_list[-1] = fr
            else:
                # Contraction
                if fr < fx_list[-1]:
                    # Outside contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                fc = func(np.clip(xc, -5, 5))
                calls += 1

                if fc <= min(fr, fx_list[-1]):
                    simplex[-1] = np.clip(xc, -5, 5)
                    fx_list[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fx_list[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break

            # Update best
            min_idx = int(np.argmin(fx_list))
            if fx_list[min_idx] < best_f:
                best_f = fx_list[min_idx]
                best_x = simplex[min_idx].copy()

            # Check for convergence or stagnation
            spread = np.max(fx_list) - np.min(fx_list)
            if spread < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
