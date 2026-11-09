"""Nelder-Mead with adaptive valley stretching for Rosenbrock f8."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex with adaptive valley stretching for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        # Initialize simplex around center
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls = 1

        # Create initial simplex
        simplex = [x0.copy()]
        fvals = [f0]
        delta = 1.0
        for i in range(dim):
            xi = x0.copy()
            xi[i] += delta
            simplex.append(xi)
            fi = func(xi)
            fvals.append(fi)
            calls += 1

        best_x = x0.copy()
        best_f = f0

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        # Valley stretching: track the direction of most recent improvement
        valley_dir = np.ones(dim) / np.sqrt(dim)
        valley_strength = 0.0

        # Adaptive step scaling for valley direction
        valley_scale = 1.0

        for iteration in range(budget - calls):
            if calls >= budget:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Track best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

            # Check convergence
            f_range = fvals[-1] - fvals[0]
            f_scale = max(abs(fvals[0]), 1e-10)
            if f_range < 1e-8 * f_scale:
                # Converged; try to escape with a random perturbation
                if calls < budget - 5:
                    x_pert = best_x + rng.normal(0, 0.1, dim)
                    x_pert = np.clip(x_pert, -5, 5)
                    f_pert = func(x_pert)
                    calls += 1
                    if f_pert < best_f:
                        best_f = f_pert
                        best_x = x_pert.copy()
                        # Reinitialize simplex around new best
                        simplex = [best_x.copy()]
                        fvals = [best_f]
                        for i in range(dim):
                            xi = best_x.copy()
                            xi[i] += 0.1
                            simplex.append(xi)
                            fi = func(xi)
                            fvals.append(fi)
                            calls += 1
                        continue
                break

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Worst point
            x_worst = simplex[-1]
            f_worst = fvals[-1]

            # Valley-adaptive reflection: stretch along valley direction
            # If we've been improving along a consistent direction, amplify it
            improvement_dir = simplex[0] - x_worst
            improvement_norm = np.linalg.norm(improvement_dir)
            if improvement_norm > 1e-10:
                improvement_dir = improvement_dir / improvement_norm
                # Blend with tracked valley direction
                valley_dir = 0.7 * improvement_dir + 0.3 * valley_dir
                valley_dir /= (np.linalg.norm(valley_dir) + 1e-10)
                valley_strength = min(valley_strength + 0.1, 1.0)
            else:
                valley_strength = max(valley_strength - 0.1, 0.0)

            # Adaptive valley scale: stretch more when in the valley
            valley_scale = 1.0 + valley_strength * 1.5

            # Reflection
            x_refl = centroid + alpha * (centroid - x_worst)
            # Add valley stretching to reflection
            x_refl = centroid + alpha * (centroid - x_worst) + \
                     valley_scale * 0.3 * valley_strength * (centroid - x_worst)
            x_refl = np.clip(x_refl, -5, 5)
            f_refl = func(x_refl)
            calls += 1

            if f_refl < fvals[0]:
                # Expansion
                x_exp = centroid + gamma * (x_refl - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls += 1
                if f_exp < f_refl:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_refl
                    fvals[-1] = f_refl
            elif f_refl < f_worst:
                simplex[-1] = x_refl
                fvals[-1] = f_refl
            else:
                # Contraction
                if f_refl < f_worst:
                    # Outside contraction
                    x_con = centroid + rho * (x_refl - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (x_worst - centroid)
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls += 1
                if f_con <= f_refl:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
