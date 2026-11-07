"""Rewritten optimizer for BBOB f13 Sharp Ridge."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f13 (Sharp Ridge):
    - Phase 1: Brief random sampling to find a good starting point
    - Phase 2: Nelder-Mead with adaptive step size and ridge-aware restarts
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        rng = np.random.default_rng(42)

        calls = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                f = float(func(x))
            except:
                return None
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Random sampling to find a good basin
        n_random = min(30, budget // 10)
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, dim)
            fx = safe_call(x)
            if fx is None:
                break

        if best_x is None:
            return best_x, best_f

        # Phase 2: Nelder-Mead with restarts
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_shrink = 0.5

        init_step = 0.5

        while calls < budget - (dim + 1):
            if time.time() > deadline:
                break

            center = best_x.copy()

            # Initialize simplex
            simplex = [center.copy()]
            f_values = [best_f]

            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                xi = center.copy()
                xi[i] += init_step
                fi = safe_call(xi)
                if fi is None:
                    break
                simplex.append(xi)
                f_values.append(fi)

            if calls >= budget or len(simplex) < dim + 1:
                break

            simplex = np.array(simplex)
            f_values = np.array(f_values)

            # Main NM loop
            stagnation_count = 0
            max_stagnation = 20

            while calls < budget:
                if time.time() > deadline:
                    break

                # Sort simplex
                idx = np.argsort(f_values)
                simplex = simplex[idx]
                f_values = f_values[idx]

                # Check for convergence
                spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                f_spread = np.max(f_values) - np.min(f_values)
                if np.max(spread) < 1e-8 or f_spread < 1e-12:
                    break

                # Centroid
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = safe_call(xr)
                if fr is None:
                    break

                improved = False
                if fr < f_values[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = safe_call(xe)
                    if fe is None:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        f_values[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_values[-1] = fr
                    improved = True
                elif fr < f_values[-2]:
                    # Reflection accepted
                    simplex[-1] = xr
                    f_values[-1] = fr
                    improved = True
                else:
                    # Contraction
                    if fr < f_values[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = safe_call(xc)
                    if fc is None:
                        break
                    if fc <= min(fr, f_values[-1]):
                        simplex[-1] = xc
                        f_values[-1] = fc
                        improved = True
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = safe_call(simplex[i])
                            if fi is None:
                                break
                            f_values[i] = fi
                        if fi is None:
                            break

                if improved:
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                if stagnation_count >= max_stagnation:
                    break

            # If we broke out due to stagnation and have budget left, restart with larger step
            if calls < budget - (dim + 1) and time.time() <= deadline:
                # Try a larger step for restart
                init_step *= 1.5
                if init_step > 2.0:
                    init_step = 2.0
                # Perturb the best point
                new_center = np.clip(best_x + rng.normal(0, 0.3, dim), -5, 5)
                new_f = safe_call(new_center)
                if new_f is None:
                    break
                # Update best if improved
                # best_x and best_f already updated in safe_call

        return best_x, best_f

# EVOLVE-BLOCK-END
