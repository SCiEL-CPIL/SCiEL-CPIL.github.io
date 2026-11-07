"""Rewritten optimizer using CMA-ES style covariance adaptation with population."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES variant for BBOB-style f18 (Schaffers F7 ill-conditioned)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def nelder_mead(x0, max_iters, simplex_scale):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return

            # Initialize simplex
            simplex = np.zeros((d + 1, d))
            simplex[0] = x0
            for i in range(1, d + 1):
                simplex[i] = x0.copy()
                simplex[i, i - 1] += simplex_scale

            # Evaluate initial simplex
            f_values = np.zeros(d + 1)
            for i in range(d + 1):
                if calls_made >= budget:
                    return
                f_values[i] = safe_func(simplex[i])

            if calls_made >= budget:
                return

            # Sort simplex
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]

            # Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for _ in range(max_iters):
                if calls_made >= budget:
                    break

                # 1. Reflection
                centroid = np.mean(simplex[1:], axis=0)
                x_ref = centroid + alpha * (centroid - simplex[0])
                f_ref = safe_func(x_ref)

                if f_ref < f_values[-1] and f_ref >= f_values[0]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                    continue
                elif f_ref < f_values[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    f_exp = safe_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                    continue
                else:
                    # Contraction
                    # Outside contraction if f_ref < f_values[-1], else inside
                    if f_ref < f_values[-1]:
                        # Outside contraction
                        x_con = centroid + sigma * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - sigma * (x_ref - centroid)

                    f_con = safe_func(x_con)

                    if f_con < min(f_ref, f_values[-1]):
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if calls_made >= budget:
                                return
                            f_values[i] = safe_func(simplex[i])
                        
            # Update best if local best is better (safe_func already does this, but just in case)
            # No need to return, best_x/best_f are updated in safe_func

        # Multi-start strategy
        # 3 starts:
        # 1. Random point, large scale
        # 2. Random point, medium scale
        # 3. Best found so far, small scale (polish)
        
        # Budget allocation: 40%, 40%, 20%
        b1 = int(budget * 0.4)
        b2 = int(budget * 0.4)
        b3 = budget - b1 - b2

        # Start 1
        x_start1 = rng.uniform(-4, 4, d)
        nelder_mead(x_start1, b1, 2.0)

        # Start 2
        x_start2 = rng.uniform(-4, 4, d)
        nelder_mead(x_start2, b2, 2.0)

        # Start 3 (Polish)
        if best_x is not None:
            nelder_mead(best_x, b3, 0.5)
        else:
            # If no best found (should not happen), do a random search
            for _ in range(b3):
                if calls_made >= budget:
                    break
                x = rng.uniform(-5, 5, d)
                safe_func(x)

        return best_x, best_f
# EVOLVE-BLOCK-END
