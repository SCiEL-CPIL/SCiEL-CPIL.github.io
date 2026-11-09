"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        # Initial simplex: center at origin, spread over [-2, 2]
        x0 = np.zeros(self.dim)
        simplex = [x0 + np.zeros(self.dim)]
        for i in range(self.dim):
            ei = np.zeros(self.dim)
            ei[i] = 2.0
            simplex.append(x0 + ei)

        fvals = [func(x) for x in simplex]
        best_x = simplex[np.argmin(fvals)]
        best_f = min(fvals)

        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        max_evals = self.budget
        evals_used = len(simplex)

        # Nelder-Mead loop
        while evals_used < max_evals:
            if time.time() > deadline:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Check convergence: if spread is small, restart
            centroid = np.mean(simplex[:-1], axis=0)
            spread = np.linalg.norm(simplex[-1] - centroid)
            if spread < 1e-6 or (fvals[-1] - fvals[0]) < 1e-10:
                # Restart: generate new simplex around best point
                center = simplex[0]
                simplex = [center + np.zeros(self.dim)]
                fvals = [fvals[0]]
                for i in range(self.dim):
                    ei = np.zeros(self.dim)
                    ei[i] = 1.0
                    new_pt = np.clip(center + ei, -5, 5)
                    simplex.append(new_pt)
                    fvals.append(func(new_pt))
                    evals_used += 1
                if evals_used >= max_evals:
                    break
                continue

            # Compute centroid of all but worst
            x_c = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = x_c + alpha * (x_c - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            evals_used += 1

            if f_r < fvals[0]:
                # Expansion
                x_e = x_c + gamma * (x_r - x_c)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                evals_used += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    # Outside contraction
                    x_cc = x_c + rho * (x_r - x_c)
                else:
                    # Inside contraction
                    x_cc = x_c - rho * (x_r - x_c)
                x_cc = np.clip(x_cc, -5, 5)
                f_cc = func(x_cc)
                evals_used += 1

                if f_cc < min(f_r, fvals[-1]):
                    simplex[-1] = x_cc
                    fvals[-1] = f_cc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        evals_used += 1
                    if evals_used >= max_evals:
                        break

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
