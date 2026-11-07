"""Optimized BBOB f6 solver using Nelder-Mead simplex with adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with multi-restart for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        total_calls = 0

        def evaluate(x):
            nonlocal total_calls, best_x, best_f
            total_calls += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(x0, budget_alloc, alpha=1.0, gamma=2.0, beta=0.5, delta=0.5):
            nonlocal total_calls, best_x, best_f
            dim = self.dim

            # Initialize simplex: x0 plus dim perturbed points
            step = 0.25
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += step
                simplex.append(np.clip(xi, -5, 5))

            fvals = [evaluate(x) for x in simplex]

            # Nelder-Mead parameters
            max_iter = 500
            stagnation_count = 0
            stagnation_limit = 30

            for _ in range(max_iter):
                if total_calls >= budget or time.time() > deadline:
                    break
                if len(simplex) < dim + 1:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check stagnation
                if fvals[0] > fvals[1] + 1e-8:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                if stagnation_count > stagnation_limit:
                    break

                x_best = simplex[0]
                x_worst = simplex[-1]
                x_second_worst = simplex[-2]

                # Centroid of all points except worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = evaluate(x_r)

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = evaluate(x_e)
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
                    x_c = centroid + beta * (x_worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = evaluate(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink all points toward best
                        for i in range(1, len(simplex)):
                            simplex[i] = x_best + delta * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = evaluate(simplex[i])

                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

            return simplex[0], fvals[0]

        # Multi-restart strategy
        # Phase 1: Nelder-Mead from center
        if total_calls < budget and time.time() < deadline:
            x0 = np.zeros(dim)
            x0 = np.clip(x0, -5, 5)
            nelder_mead(x0, int(budget * 0.4))

        # Phase 2: Nelder-Mead from random point
        if total_calls < budget and time.time() < deadline:
            x0 = rng.uniform(-3, 3, dim)
            x0 = np.clip(x0, -5, 5)
            nelder_mead(x0, int(budget * 0.3))

        # Phase 3: Nelder-Mead from another random point
        if total_calls < budget and time.time() < deadline:
            x0 = rng.uniform(-4, 4, dim)
            x0 = np.clip(x0, -5, 5)
            nelder_mead(x0, int(budget * 0.2))

        # Phase 4: Final local refinement from best found
        if best_x is not None and total_calls < budget and time.time() < deadline:
            remaining = budget - total_calls
            if remaining > dim + 1:
                nelder_mead(best_x.copy(), remaining)

        # Ensure valid result
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
