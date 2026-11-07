"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with center init and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        x = np.zeros(self.dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx
        calls = 1

        # Initialize simplex: center + small perturbations along each axis
        # and one point with a larger initial step to help escape flat regions
        step = 0.5
        simplex = [x.copy()]
        for i in range(self.dim):
            xi = x.copy()
            xi[i] += step
            simplex.append(xi)
        # One extra point with a bigger step in a diagonal direction
        xd = x.copy()
        xd += 0.3 * np.ones(self.dim)
        simplex.append(xd)

        # Evaluate all simplex points
        fx_list = [fx]
        for i in range(1, len(simplex)):
            if time.time() > deadline:
                break
            xi = np.clip(simplex[i], -5, 5)
            fi = func(xi)
            simplex[i] = xi
            fx_list.append(fi)
            calls += 1
            if fi < best_fx:
                best_fx, best_x = fi, xi.copy()

        n_simplex = len(simplex)

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        while calls < self.budget:
            if time.time() > deadline:
                break

            # Sort simplex by function value
            order = np.argsort(fx_list)
            simplex = [simplex[i] for i in order]
            fx_list = [fx_list[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_fx = fx_list[-1]
            best_fx_simplex = fx_list[0]

            # Reflection
            xr = centroid + alpha * (centroid - worst)
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            if fr < best_fx_simplex:
                # Try expansion
                xe = centroid + gamma * (centroid - worst)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    fx_list[-1] = fe
                    if fe < best_fx:
                        best_fx, best_x = fe, xe.copy()
                else:
                    simplex[-1] = xr
                    fx_list[-1] = fr
                    if fr < best_fx:
                        best_fx, best_x = fr, xr.copy()
            elif fr < worst_fx:
                simplex[-1] = xr
                fx_list[-1] = fr
                if fr < best_fx:
                    best_fx, best_x = fr, xr.copy()
            else:
                # Contraction
                if fr < worst_fx:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc <= fr:
                        simplex[-1] = xc
                        fx_list[-1] = fc
                        if fc < best_fx:
                            best_fx, best_x = fc, xc.copy()
                    else:
                        # Shrink
                        for i in range(1, n_simplex):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            calls += 1
                            fx_list[i] = fi
                            if fi < best_fx:
                                best_fx, best_x = fi, simplex[i].copy()
                else:
                    # Inside contraction
                    xc = centroid - rho * (worst - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < worst_fx:
                        simplex[-1] = xc
                        fx_list[-1] = fc
                        if fc < best_fx:
                            best_fx, best_x = fc, xc.copy()
                    else:
                        # Shrink
                        for i in range(1, n_simplex):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            calls += 1
                            fx_list[i] = fi
                            if fi < best_fx:
                                best_fx, best_x = fi, simplex[i].copy()

            # Check for convergence: simplex is very small
            spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            if np.all(spread < 1e-8):
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
