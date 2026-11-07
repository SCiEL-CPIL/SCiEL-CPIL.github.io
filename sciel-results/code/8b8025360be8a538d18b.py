"""Rewritten optimizer: Differential Evolution with adaptive mutation + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """DE/rand/1 with adaptive F and Nelder-Mead final polish."""

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

        # DE/rand/1 with adaptive mutation factor
        def differential_evolution(x_init, max_evals, rng, func):
            pop_size = max(20, 4 * d)
            # Initialize population around x_init with some spread
            spread = 2.0
            pop = np.array([np.clip(x_init + spread * rng.standard_normal(d), -5, 5)
                           for _ in range(pop_size)])
            # Evaluate initial population
            fits = np.array([func(x) for x in pop])
            calls = pop_size
            best_idx = np.argmin(fits)
            best_x = pop[best_idx].copy()
            best_f = fits[best_idx]

            # Adaptive mutation factor
            F = 0.8
            CR = 0.5
            no_improve = 0

            while calls < max_evals:
                if time_exceeded():
                    break

                # Generate trial vector for each individual
                for i in range(pop_size):
                    if calls >= max_evals:
                        break
                    if time_exceeded():
                        break

                    # Select three distinct random indices different from i
                    while True:
                        a = rng.integers(pop_size)
                        if a != i:
                            break
                    while True:
                        b = rng.integers(pop_size)
                        if b != i and b != a:
                            break
                    while True:
                        c = rng.integers(pop_size)
                        if c != i and c != a and c != b:
                            break

                    # Mutation: DE/rand/1
                    v = pop[i] + F * (pop[a] - pop[b]) + F * (pop[b] - pop[c])
                    v = np.clip(v, -5, 5)

                    # Crossover: binomial
                    j_rand = rng.integers(d)
                    trial = pop[i].copy()
                    for j in range(d):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]

                    fy = func(trial)
                    calls += 1

                    # Selection
                    if fy <= fits[i]:
                        pop[i] = trial
                        fits[i] = fy
                        if fy < best_f:
                            best_f = fy
                            best_x = trial.copy()
                            no_improve = 0
                    else:
                        no_improve += 1

                # Adapt F: decrease if improving, increase if stagnating
                if no_improve > pop_size * 2:
                    F = min(0.95, F + 0.1)
                    no_improve = 0
                elif no_improve < pop_size // 4:
                    F = max(0.4, F - 0.05)

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

        # Phase 1: Initial sampling to find good starting points
        n_initial = min(20, budget // 15)
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

        # Phase 2: DE from multiple starting points
        remaining = budget - calls_used
        n_de_runs = min(3, max(1, remaining // 80))
        de_budget = remaining // n_de_runs

        for run in range(n_de_runs):
            if time_exceeded() or calls_used >= budget:
                break

            if run == 0:
                x0 = candidates[0][0].copy()
            elif run < len(candidates):
                x0 = candidates[run][0].copy()
            else:
                x0 = best_x + rng.standard_normal(d) * 0.5
                x0 = np.clip(x0, -5, 5)

            new_x, new_f, calls_made = differential_evolution(
                x0, min(de_budget, budget - calls_used), rng, func)
            calls_used += calls_made

            if new_f < best_f:
                best_f = new_f
                best_x = new_x.copy()

        # Phase 3: Nelder-Mead polish
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
