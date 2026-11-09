"""DE + Nelder-Mead hybrid optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """DE for global search + Nelder-Mead for local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        x_best = None
        f_best = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal x_best, f_best, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        def nelder_mead(x0, max_calls, initial_step=0.5):
            """Nelder-Mead refinement from x0."""
            nonlocal x_best, f_best, calls_made
            if calls_made >= max_calls:
                return

            # Build initial simplex
            simplex = [np.array(x0, dtype=np.float64)]
            for d in range(dim):
                pert = np.array(x0, dtype=np.float64)
                pert[d] += initial_step
                simplex.append(np.clip(pert, -5, 5))

            fvals = []
            for v in simplex:
                if not time_ok() or calls_made >= max_calls:
                    return
                fvals.append(evaluate(v))

            # Nelder-Mead iterations
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            max_iter = max(1, (max_calls - calls_made) // (dim + 1))

            for _ in range(max_iter):
                if not time_ok() or calls_made >= max_calls:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
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
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = evaluate(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = evaluate(simplex[i])

        def de_search(pop_size, max_calls, f1=0.8, cr=0.9):
            """Differential Evolution search."""
            nonlocal x_best, f_best, calls_made

            # Initialize population
            rng = np.random.default_rng(42)
            pop = np.array([rng.uniform(-4.5, 4.5, dim) for _ in range(pop_size)])
            pop = np.clip(pop, -5, 5)
            fpop = np.array([evaluate(p) for p in pop])

            best_idx = np.argmin(fpop)
            pop_best = pop[best_idx].copy()
            f_best_de = fpop[best_idx]

            no_improve = 0
            max_gen = max(1, (max_calls - calls_made) // pop_size)

            for gen in range(max_gen):
                if not time_ok() or calls_made >= max_calls:
                    break

                for i in range(pop_size):
                    if not time_ok() or calls_made >= max_calls:
                        break

                    # Select three distinct random individuals
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                    while r1 == r2 or r1 == i or r2 == i:
                        r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)

                    # Mutation
                    v = pop[r1] + f1 * (pop[r2] - pop[r3])
                    v = np.clip(v, -5, 5)

                    # Crossover
                    trial = np.array(pop[i])
                    j_rand = rng.integers(dim)
                    for j in range(dim):
                        if rng.random() < cr or j == j_rand:
                            trial[j] = v[j]
                    trial = np.clip(trial, -5, 5)

                    # Selection
                    f_trial = evaluate(trial)
                    if f_trial <= fpop[i]:
                        pop[i] = trial
                        fpop[i] = f_trial

                    if f_trial < f_best_de:
                        f_best_de = f_trial
                        pop_best = trial.copy()
                        no_improve = 0
                    else:
                        no_improve += 1

                # Adapt parameters based on stagnation
                if no_improve > 2 * pop_size:
                    f1 = max(0.4, f1 * 0.9)
                    cr = min(0.95, cr * 1.05)
                    no_improve = 0

                # Inject best found so far
                if x_best is not None and f_best < f_best_de:
                    pop_best = x_best.copy()
                    f_best_de = f_best

            return pop_best, f_best_de

        # Main strategy:
        # Phase 1: Quick grid to find a good starting point
        # Phase 2: DE for global search
        # Phase 3: Nelder-Mead for local refinement
        # Phase 4: Repeat DE + NM cycles if budget remains

        # Phase 1: Coarse grid (small, just to seed DE)
        grid_values = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
        grid_points = list(np.meshgrid(*([grid_values] * dim), indexing='ij'))
        grid_points = np.array(grid_points).reshape(dim, -1).T

        # Evaluate a subset of grid points (not all 3125)
        rng_init = np.random.default_rng(0)
        subset_size = min(100, len(grid_points))
        indices = rng_init.choice(len(grid_points), size=subset_size, replace=False)
        for idx in indices:
            if not time_ok():
                break
            evaluate(grid_points[idx])

        # Phase 2: DE search
        pop_size = 20
        de_budget = int(budget * 0.5)
        de_max_calls = calls_made + de_budget
        if time_ok() and de_budget >= pop_size * 2:
            de_search(pop_size, de_max_calls)

        # Phase 3: Nelder-Mead from best
        if x_best is not None and time_ok():
            nm_budget = min(budget - calls_made, int(budget * 0.2))
            if nm_budget >= dim + 2:
                nelder_mead(x_best.copy(), calls_made + nm_budget, initial_step=0.5)

        # Phase 4: Second DE round with smaller population, focused around best
        if time_ok() and budget - calls_made >= 30:
            rng2 = np.random.default_rng(123)
            pop_size2 = 15
            remaining = budget - calls_made
            if remaining >= pop_size2 * 3:
                # Initialize around current best
                center = x_best.copy() if x_best is not None else np.zeros(dim)
                spread = 1.5
                pop2 = np.array([np.clip(center + rng2.normal(0, spread, dim), -5, 5)
                                 for _ in range(pop_size2)])
                fpop2 = np.array([evaluate(p) for p in pop2])

                best_idx2 = np.argmin(fpop2)
                pop2_best = pop2[best_idx2].copy()
                f_best2 = fpop2[best_idx2]
                no_improve2 = 0
                max_gen2 = max(1, (budget - calls_made) // pop_size2)

                for gen in range(max_gen2):
                    if not time_ok() or calls_made >= budget:
                        break
                    for i in range(pop_size2):
                        if not time_ok() or calls_made >= budget:
                            break
                        r1, r2, r3 = rng2.choice(pop_size2, size=3, replace=False)
                        while r1 == r2 or r1 == i or r2 == i:
                            r1, r2, r3 = rng2.choice(pop_size2, size=3, replace=False)
                        v = pop2[r1] + 0.7 * (pop2[r2] - pop2[r3])
                        v = np.clip(v, -5, 5)
                        trial = np.array(pop2[i])
                        j_rand = rng2.integers(dim)
                        for j in range(dim):
                            if rng2.random() < 0.8 or j == j_rand:
                                trial[j] = v[j]
                        trial = np.clip(trial, -5, 5)
                        f_trial = evaluate(trial)
                        if f_trial <= fpop2[i]:
                            pop2[i] = trial
                            fpop2[i] = f_trial
                        if f_trial < f_best2:
                            f_best2 = f_trial
                            pop2_best = trial.copy()
                            no_improve2 = 0
                        else:
                            no_improve2 += 1

        # Phase 5: Final Nelder-Mead refinement
        if x_best is not None and time_ok() and budget - calls_made >= dim + 2:
            nelder_mead(x_best.copy(), budget, initial_step=0.3)

        # Final safety
        if x_best is None:
            x_best = np.zeros(dim)
            f_best = evaluate(x_best)

        return x_best, f_best
# EVOLVE-BLOCK-END
