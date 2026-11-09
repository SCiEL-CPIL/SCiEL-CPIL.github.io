"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart Nelder-Mead with adaptive step sizes and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            calls_made += 1
            return func(x)

        def nelder_mead(x0, max_iters):
            nonlocal best_x, best_f, calls_made
            n = dim
            # Create simplex with adaptive initial step
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                # Use smaller initial step for better convergence
                step = 0.3 if abs(x0[i]) < 4.7 else 0.1
                simplex[i + 1][i] += step if x0[i] < 4.7 else -step

            # Evaluate initial simplex
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                fx = eval_func(simplex[i])
                if fx is None:
                    return
                fvals[i] = fx
                if fx < best_f:
                    best_f = fx
                    best_x = simplex[i].copy()

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            min_step = 1e-12

            for _ in range(max_iters):
                if time.time() > deadline:
                    return
                if calls_made >= budget - 2:
                    return

                # Sort simplex by function value
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = eval_func(xr)
                if fr is None:
                    return
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = eval_func(xe)
                    if fe is None:
                        return
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                        if fe < best_f:
                            best_f = fe
                            best_x = xe.copy()
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                        fc = eval_func(xc)
                        if fc is None:
                            return
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                            if fc < best_f:
                                best_f = fc
                                best_x = xc.copy()
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                fvals[i] = eval_func(simplex[i])
                                if fvals[i] is None:
                                    return
                                if fvals[i] < best_f:
                                    best_f = fvals[i]
                                    best_x = simplex[i].copy()
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (centroid - simplex[-1])
                        fc = eval_func(xc)
                        if fc is None:
                            return
                        if fc <= fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                            if fc < best_f:
                                best_f = fc
                                best_x = xc.copy()
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                fvals[i] = eval_func(simplex[i])
                                if fvals[i] is None:
                                    return
                                if fvals[i] < best_f:
                                    best_f = fvals[i]
                                    best_x = simplex[i].copy()

                # Check convergence
                if np.max(fvals) - np.min(fvals) < min_step:
                    return

        def local_es(x_center, n_evals):
            """Simple evolutionary strategy for local refinement."""
            nonlocal best_x, best_f, calls_made
            n = dim
            pop_size = min(n_evals // 2, 20)
            if pop_size < 4:
                pop_size = 4
            sigma = 0.5

            # Initial population
            pop = [x_center.copy()]
            for _ in range(pop_size - 1):
                x_new = x_center + sigma * np.random.randn(n)
                x_new = np.clip(x_new, -5, 5)
                pop.append(x_new)

            pop_f = []
            for x in pop:
                fx = eval_func(x)
                if fx is None:
                    return
                pop_f.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            for gen in range(max(1, n_evals // pop_size)):
                if time.time() > deadline:
                    return
                if calls_made >= budget - 2:
                    return

                # Sort by fitness
                order = np.argsort(pop_f)
                pop = [pop[i] for i in order]
                pop_f = [pop_f[i] for i in order]

                # Create new population
                new_pop = [pop[0].copy()]
                for _ in range(pop_size - 1):
                    parent_idx = int(np.random.choice(pop_size, p=np.array([1.0/(i+1) for i in range(pop_size)])))
                    x_new = pop[parent_idx] + sigma * np.random.randn(n)
                    x_new = np.clip(x_new, -5, 5)
                    new_pop.append(x_new)

                new_pop_f = []
                for x in new_pop:
                    fx = eval_func(x)
                    if fx is None:
                        return
                    new_pop_f.append(fx)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                # Combine and select
                combined_pop = pop[:pop_size//2] + new_pop
                combined_f = pop_f[:pop_size//2] + new_pop_f
                order = np.argsort(combined_f)
                pop = [combined_pop[i] for i in order[:pop_size]]
                pop_f = [combined_f[i] for i in order[:pop_size]]

                # Adapt sigma
                improvement = pop_f[0] - pop_f[-1]
                if improvement > 1e-6:
                    sigma *= 0.8
                else:
                    sigma *= 1.2
                sigma = np.clip(sigma, 0.01, 5.0)

        # Multi-restart strategy
        # First, do a coarse random search to find good starting points
        rng = np.random.default_rng(42)

        # Phase 1: Random sampling to find promising regions
        n_random = 50
        for i in range(n_random):
            if time.time() > deadline:
                break
            if calls_made >= budget - 10:
                break
            x = rng.uniform(-5, 5, dim)
            fx = eval_func(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Nelder-Mead from best point
        if best_x is not None and calls_made < budget - 10:
            nelder_mead(best_x.copy(), max_iters=200)

        # Phase 3: More Nelder-Mead restarts from diverse points
        # Generate diverse starting points around the best
        n_restarts = 5
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if calls_made >= budget - 10:
                break

            # Create a perturbed start around best
            scale = 2.0 * (0.5 ** r)
            x_start = best_x + scale * rng.standard_normal(dim)
            x_start = np.clip(x_start, -5, 5)

            nelder_mead(x_start, max_iters=100)

        # Phase 4: Local ES refinement if budget remains
        if best_x is not None and calls_made < budget - 2:
            remaining = budget - calls_made - 2
            local_es(best_x.copy(), min(remaining, 100))

        # Phase 5: Final local refinement
        if best_x is not None and calls_made < budget - 2:
            nelder_mead(best_x.copy(), max_iters=50)

        return best_x, best_f
# EVOLVE-BLOCK-END
