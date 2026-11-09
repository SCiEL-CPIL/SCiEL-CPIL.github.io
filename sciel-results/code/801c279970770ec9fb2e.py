# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead with coordinate-descent warm start and adaptive restarts for f13 Sharp Ridge."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > budget:
                raise Exception("Budget exceeded")
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Coordinate-descent warm start for early progress
        # This gives quick initial improvement for AOCC
        x = np.zeros(n)
        fx = wrapped_func(x)

        # Simple coordinate descent for a few steps
        cd_steps = 10
        for step in range(cd_steps):
            if calls_made[0] >= budget:
                break
            improved = False
            for i in range(n):
                if calls_made[0] >= budget:
                    break
                # Try moving in each direction
                for delta in [0.5, -0.5, 0.2, -0.2, 0.1, -0.1]:
                    if calls_made[0] >= budget:
                        break
                    trial = x.copy()
                    trial[i] += delta
                    trial = np.clip(trial, -5, 5)
                    ftrial = wrapped_func(trial)
                    if ftrial < fx:
                        x, fx = trial, ftrial
                        improved = True
                        break
            if not improved:
                break

        # Phase 2: Nelder-Mead optimization
        def nelder_mead(x0, f0, remaining_budget, max_iter):
            """Nelder-Mead simplex optimization."""
            nonlocal calls_made, best_f, best_x
            if calls_made[0] >= budget:
                return x0, f0

            # Initialize simplex
            alpha = 0.5
            simplex = [x0.copy()]
            fvals = [f0]
            for i in range(n):
                if calls_made[0] >= budget:
                    break
                xi = x0.copy()
                if xi[i] != 0:
                    xi[i] += alpha * abs(xi[i])
                else:
                    xi[i] += alpha
                xi = np.clip(xi, -5, 5)
                fi = wrapped_func(xi)
                simplex.append(xi)
                fvals.append(fi)

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            x_best = simplex[0].copy()
            f_best = fvals[0]

            for it in range(max_iter):
                if calls_made[0] >= budget:
                    break
                if time.time() > deadline:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if fvals[0] < f_best:
                    f_best = fvals[0]
                    x_best = simplex[0].copy()

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = wrapped_func(x_r)
                if calls_made[0] >= budget:
                    break

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + 2 * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = wrapped_func(x_e)
                    if calls_made[0] >= budget:
                        break
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
                        x_c = centroid + 0.5 * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - 0.5 * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = wrapped_func(x_c)
                    if calls_made[0] >= budget:
                        break
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            if calls_made[0] >= budget:
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = wrapped_func(simplex[i])

            return x_best, f_best

        # Run Nelder-Mead with remaining budget
        remaining = budget - calls_made[0]
        if remaining > 0:
            # Use most of the budget for Nelder-Mead
            max_iter = min(remaining - 1, 400)  # Leave 1 call for safety
            x, fx = nelder_mead(x, fx, remaining, max_iter)

        # Phase 3: Polish with fine coordinate descent near the best point
        if calls_made[0] < budget and time.time() < deadline:
            x_polish = best_x.copy()
            f_polish = best_f
            step_size = 0.1
            for _ in range(50):
                if calls_made[0] >= budget or time.time() > deadline:
                    break
                improved = False
                for i in range(n):
                    if calls_made[0] >= budget:
                        break
                    for delta in [step_size, -step_size]:
                        if calls_made[0] >= budget:
                            break
                        trial = x_polish.copy()
                        trial[i] += delta
                        trial = np.clip(trial, -5, 5)
                        ftrial = wrapped_func(trial)
                        if ftrial < f_polish:
                            x_polish, f_polish = trial, ftrial
                            improved = True
                            break
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-4:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
