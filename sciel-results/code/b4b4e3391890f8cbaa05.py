"""Optimizer for BBOB f17 (Schaffer's F7) using Differential Evolution with L-BFGS-B refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Differential Evolution with L-BFGS-B local refinement for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, leave 15s margin

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            fx = float(func(x))
            calls_made += 1
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        def lbfgsb_refine(x0, max_calls):
            """L-BFGS-B local refinement using finite-difference gradients."""
            nonlocal best_x, best_fx, calls_made
            if not time_ok() or calls_made >= max_calls:
                return best_x, best_fx

            # Count how many calls we can afford (L-BFGS-B uses ~d+1 per gradient eval)
            remaining = max_calls - calls_made
            if remaining < 5:
                return best_x, best_fx

            # Use a wrapper that tracks calls
            def objective(x):
                nonlocal calls_made
                if calls_made >= max_calls:
                    return np.inf
                fx = float(func(x))
                calls_made += 1
                return fx

            try:
                res = minimize(
                    objective, x0, method='L-BFGS-B',
                    bounds=[(-5, 5)] * d,
                    options={'maxiter': max(1, (max_calls - calls_made) // (d + 1)),
                             'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

            return best_x, best_fx

        def differential_evolution(pop, fvals, max_calls, F_init=0.8, CR=0.7):
            """DE/rand/1/bin with adaptive parameters."""
            nonlocal best_x, best_fx, calls_made
            n = d
            NP = len(pop)
            if NP < 4:
                return best_x, best_fx

            # Adaptive F: start high, decrease over time for convergence
            # CR: start moderate, can increase for diversity

            pop = [x.copy() for x in pop]
            fvals = list(fvals)

            idx = np.argmin(fvals)
            if fvals[idx] < best_fx:
                best_fx = fvals[idx]
                best_x = pop[idx].copy()

            gen = 0
            max_gen = max(1, (max_calls - calls_made) // NP)

            for gen in range(max_gen):
                if not time_ok() or calls_made >= max_calls:
                    break

                # Adaptive parameters
                progress = gen / max(1, max_gen - 1)
                F = F_init * (1.0 - 0.5 * progress) + 0.3 * progress  # 0.8 -> 0.5
                CR_eff = CR * (1.0 - 0.3 * progress) + 0.9 * progress  # 0.7 -> 0.85

                # Sort by fitness
                order = np.argsort(fvals)
                pop = [pop[i] for i in order]
                fvals = [fvals[i] for i in order]

                new_pop = [pop[0].copy()]
                new_fvals = [fvals[0]]

                for i in range(1, NP):
                    if not time_ok() or calls_made >= max_calls:
                        break

                    # DE/rand/1/bin mutation
                    r1, r2, r3 = rng.choice(NP, 3, replace=False)
                    while r1 == r2 or r1 == r3 or r2 == r3:
                        r1, r2, r3 = rng.choice(NP, 3, replace=False)

                    v = pop[r1] + F * (pop[r2] - pop[r3])
                    v = np.clip(v, -5, 5)

                    # Binomial crossover
                    j_rand = rng.integers(d)
                    trial = pop[i].copy()
                    for j in range(d):
                        if rng.random() < CR_eff or j == j_rand:
                            trial[j] = v[j]
                    trial = np.clip(trial, -5, 5)

                    f_trial = call_func(trial)
                    if f_trial <= fvals[i]:
                        new_pop.append(trial)
                        new_fvals.append(f_trial)
                    else:
                        new_pop.append(pop[i])
                        new_fvals.append(fvals[i])

                # Complete with remaining population if we broke early
                while len(new_pop) < NP:
                    i = len(new_pop)
                    new_pop.append(pop[i])
                    new_fvals.append(fvals[i])

                pop = new_pop
                fvals = new_fvals

                idx = np.argmin(fvals)
                if fvals[idx] < best_fx:
                    best_fx = fvals[idx]
                    best_x = pop[idx].copy()

                # Check convergence
                if fvals[-1] - fvals[0] < 1e-12:
                    break

            return best_x, best_fx

        # Phase 1: Structured initial sampling for early progress
        points = [np.zeros(d)]
        # Axis-aligned points
        for i in range(d):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                x = np.zeros(d)
                x[i] = val
                points.append(x.copy())
        # Diagonal points
        for val in [-3.0, -1.5, 1.5, 3.0]:
            points.append(np.full(d, val))
        # Random points
        for _ in range(12):
            points.append(rng.uniform(-4.5, 4.5, d))

        for x in points:
            if not time_ok():
                break
            x = np.clip(x, -5, 5)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)

        # Phase 2: Build initial DE population from best + diverse points
        NP = 20  # population size
        pop = [best_x.copy()]
        fvals = [best_fx]

        # Add diverse points around best
        for _ in range(NP - 1):
            if not time_ok():
                break
            # Mix of local and global exploration
            if rng.random() < 0.5:
                x = best_x + rng.normal(0, 2.0, d)
            else:
                x = rng.uniform(-4.5, 4.5, d)
            x = np.clip(x, -5, 5)
            fx = call_func(x)
            pop.append(x)
            fvals.append(fx)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 3: Main DE search
        de_calls_remaining = budget - calls_made - 40  # reserve for refinement
        if de_calls_remaining > NP:
            max_calls = calls_made + de_calls_remaining
            differential_evolution(pop, fvals, max_calls, F_init=0.8, CR=0.7)

        # Phase 4: L-BFGS-B refinement on best solution
        if time_ok() and calls_made < budget:
            refine_calls = min(60, budget - calls_made)
            lbfgsb_refine(best_x.copy(), calls_made + refine_calls)

        # Phase 5: Second DE round from refined best with tighter population
        if time_ok() and calls_made < budget - 20:
            NP2 = 12
            pop2 = [best_x.copy()]
            fvals2 = [best_fx]
            for _ in range(NP2 - 1):
                if not time_ok():
                    break
                x = best_x + rng.normal(0, 1.0, d)
                x = np.clip(x, -5, 5)
                fx = call_func(x)
                pop2.append(x)
                fvals2.append(fx)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

            remaining = budget - calls_made - 20
            if remaining > NP2:
                max_calls2 = calls_made + remaining
                differential_evolution(pop2, fvals2, max_calls2, F_init=0.6, CR=0.8)

        # Phase 6: Final L-BFGS-B polish
        if time_ok() and calls_made < budget:
            refine_calls = min(30, budget - calls_made)
            lbfgsb_refine(best_x.copy(), calls_made + refine_calls)

        return best_x, best_fx
# EVOLVE-BLOCK-END
