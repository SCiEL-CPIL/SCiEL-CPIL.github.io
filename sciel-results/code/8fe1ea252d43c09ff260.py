"""Rewritten optimizer: (1+1)-ES with diverse restarts and final local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with diverse restarts and final local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        def time_exceeded():
            return time.time() > deadline

        # (1+1)-ES with adaptive step size
        def es_11(x0, max_evals, rng, func):
            x = x0.copy()
            fx = func(x)
            calls = 1
            best_x = x.copy()
            best_f = fx

            # Adaptive step size
            sigma = 1.0
            # Success rate tracking
            successes = 0
            failures = 0
            window = 20

            while calls < max_evals:
                if time_exceeded():
                    break

                # Generate mutant
                step = sigma * rng.standard_normal(d)
                x_new = x + step
                x_new = np.clip(x_new, -5, 5)
                fx_new = func(x_new)
                calls += 1

                if fx_new < fx:
                    # Success: accept mutant, increase sigma slightly
                    x = x_new
                    fx = fx_new
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Increase sigma on success
                    sigma *= 1.05
                else:
                    # Failure: keep current, decrease sigma slightly
                    failures += 1
                    # Decrease sigma on failure
                    sigma *= 0.95

                # Clamp sigma
                sigma = np.clip(sigma, 1e-4, 10.0)

                # Adapt sigma based on success rate over window
                if successes + failures >= window:
                    success_rate = successes / (successes + failures)
                    if success_rate > 0.35:
                        sigma *= 1.1
                    elif success_rate < 0.15:
                        sigma *= 0.9
                    successes = 0
                    failures = 0

                # Check convergence
                if sigma < 1e-5:
                    break

            return best_x, best_f, calls

        # Nelder-Mead polish
        def nelder_mead_polish(x0, max_evals, rng, func):
            x = x0.copy()
            fx = func(x)
            calls = 1
            best_x = x.copy()
            best_f = fx

            # Initial simplex
            alpha = 0.05
            simplex = [x.copy()]
            for i in range(d):
                xi = x.copy()
                xi[i] += alpha if xi[i] < 5 else -alpha
                simplex.append(xi)
            simplex = np.array(simplex)
            simplex_fits = np.array([func(s) for s in simplex])
            calls += d

            # Nelder-Mead parameters
            alpha_nm = 1.0  # reflection
            gamma = 2.0     # expansion
            rho = 0.5       # contraction
            sigma_nm = 0.5  # shrink

            while calls < max_evals:
                if time_exceeded():
                    break

                # Sort simplex
                order = np.argsort(simplex_fits)
                simplex = simplex[order]
                simplex_fits = simplex_fits[order]

                # Update best
                if simplex_fits[0] < best_f:
                    best_f = simplex_fits[0]
                    best_x = simplex[0].copy()

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha_nm * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1

                if fr < simplex_fits[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_fits[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_fits[-1] = fr
                elif fr < simplex_fits[-2]:
                    simplex[-1] = xr
                    simplex_fits[-1] = fr
                else:
                    # Contraction
                    if fr < simplex_fits[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)
                        calls += 1
                        if fc <= fr:
                            simplex[-1] = xc
                            simplex_fits[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, d + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                simplex_fits[i] = func(simplex[i])
                                calls += 1
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)
                        calls += 1
                        if fc < simplex_fits[-1]:
                            simplex[-1] = xc
                            simplex_fits[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, d + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                simplex_fits[i] = func(simplex[i])
                                calls += 1

                # Check convergence
                if np.max(simplex_fits) - np.min(simplex_fits) < 1e-10:
                    break

            return best_x, best_f, calls

        # Phase 1: Initial random sampling to find diverse starting points
        n_initial = min(30, budget // 10)
        candidates = []
        for i in range(n_initial):
            if time_exceeded():
                break
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            candidates.append((x, fx))

        candidates.sort(key=lambda t: t[1])
        best_x, best_f = candidates[0]
        calls_used = len(candidates)

        # Phase 2: (1+1)-ES from multiple diverse starting points
        remaining = budget - calls_used
        n_es_runs = min(5, max(1, remaining // 40))
        es_budget = remaining // n_es_runs

        for run in range(n_es_runs):
            if time_exceeded() or calls_used >= budget:
                break

            if run == 0:
                x0 = candidates[0][0].copy()
            elif run < len(candidates):
                x0 = candidates[run][0].copy()
            else:
                x0 = best_x + rng.standard_normal(d) * 0.5
                x0 = np.clip(x0, -5, 5)

            new_x, new_f, calls_made = es_11(
                x0, min(es_budget, budget - calls_used), rng, func)
            calls_used += calls_made

            if new_f < best_f:
                best_f = new_f
                best_x = new_x.copy()

        # Phase 3: Nelder-Mead polish on the best point
        remaining = budget - calls_used
        if remaining > 10 and not time_exceeded():
            new_x, new_f, calls_made = nelder_mead_polish(
                best_x, min(remaining, 50), rng, func)
            calls_used += calls_made

            if new_f < best_f:
                best_f = new_f
                best_x = new_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
