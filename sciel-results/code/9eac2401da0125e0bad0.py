"""Combined DE + Nelder-Mead + (1+1)-ES for Rastrigin-rotated f15."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-phase optimizer for Rastrigin-rotated f15:
    1. Dense grid of 25 diverse starting points with Nelder-Mead (20 evals each)
       to identify the most promising local basin
    2. Differential Evolution for global multimodal search
    3. (1+1)-ES polish with adaptive step size
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        calls_made = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_check():
            return time.time() < deadline and calls_made < budget

        def nelder_mead(x0, max_evals=20):
            """Run Nelder-Mead from x0 for at most max_evals function evaluations."""
            nonlocal calls_made, best_f, best_x
            if not time_check():
                return best_x.copy(), best_f
            remaining = budget - calls_made
            if remaining < 10:
                return best_x.copy(), best_f
            try:
                result = minimize(
                    lambda x: eval_func(x),
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max(1, max_evals // 6),
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
            except Exception:
                pass
            return best_x.copy(), best_f

        def differential_evolution(x_init, max_evals=100):
            """Simple differential evolution for global search."""
            nonlocal calls_made, best_f, best_x
            n = dim
            pop_size = max(10, 2 * n)
            F = 0.8  # scale factor
            CR = 0.9  # crossover rate

            # Initialize population around x_init and randomly
            pop = []
            pop_f = []
            # Add x_init as first individual
            x0 = np.clip(x_init, -5.0, 5.0)
            f0 = eval_func(x0)
            pop.append(x0.copy())
            pop_f.append(f0)

            for _ in range(pop_size - 1):
                if not time_check():
                    break
                x = rng.uniform(-5.0, 5.0, n)
                f = eval_func(x)
                pop.append(x.copy())
                pop_f.append(f)

            if len(pop) < 3:
                return best_x.copy(), best_f

            # Track best
            local_best_f = min(pop_f)
            local_best_x = pop[np.argmin(pop_f)].copy()

            gen = 0
            max_gen = max(1, max_evals // pop_size)
            while gen < max_gen and time_check():
                if calls_made >= budget:
                    break
                new_pop = list(pop)
                new_pop_f = list(pop_f)

                for i in range(len(pop)):
                    if not time_check():
                        break
                    if calls_made >= budget:
                        break

                    # Select three distinct random individuals
                    a, b, c = rng.choice(len(pop), size=3, replace=False)
                    if a == i or b == i or c == i:
                        continue

                    # Mutation
                    mutant = pop[a] + F * (pop[b] - pop[c])
                    mutant = np.clip(mutant, -5.0, 5.0)

                    # Crossover
                    trial = pop[i].copy()
                    j_rand = rng.integers(n)
                    for j in range(n):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = mutant[j]
                    trial = np.clip(trial, -5.0, 5.0)

                    # Evaluate
                    f_trial = eval_func(trial)

                    # Selection
                    if f_trial <= new_pop_f[i]:
                        new_pop[i] = trial
                        new_pop_f[i] = f_trial
                        if f_trial < local_best_f:
                            local_best_f = f_trial
                            local_best_x = trial.copy()

                pop = new_pop
                pop_f = new_pop_f
                gen += 1

            return local_best_x, local_best_f

        def es_polish(x0, max_evals=50):
            """(1+1)-ES with adaptive step size."""
            nonlocal calls_made, best_f, best_x
            x = x0.copy()
            fx = eval_func(x)
            local_best_f = fx
            local_best_x = x.copy()
            sigma = 0.3

            for _ in range(max_evals):
                if not time_check():
                    break
                if calls_made >= budget:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = eval_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                    if fy < local_best_f:
                        local_best_f = fy
                        local_best_x = x.copy()
                else:
                    sigma *= 0.85
                if sigma < 1e-6:
                    break

            return local_best_x, local_best_f

        # Phase 1: Dense grid of 25 diverse starting points with Nelder-Mead
        # Generate stratified starting points
        start_points = []
        # Include origin and corners of the domain
        start_points.append(np.zeros(dim))
        for _ in range(24):
            p = rng.uniform(-4.5, 4.5, dim)
            start_points.append(p)

        # Ensure we have enough budget for all 25 Nelder-Mead runs
        nm_evals_each = 20
        if len(start_points) * nm_evals_each > budget * 0.4:
            nm_evals_each = max(10, int(budget * 0.4 / len(start_points)))

        for sp in start_points:
            if not time_check():
                break
            if calls_made >= budget * 0.4:
                break
            bx, bf = nelder_mead(sp.copy(), max_evals=nm_evals_each)
            # best_f and best_x are updated globally by eval_func

        # Phase 2: Differential Evolution for global multimodal search
        if time_check() and best_f < np.inf:
            remaining = budget - calls_made
            if remaining > 50:
                de_evals = min(150, remaining)
                bx, bf = differential_evolution(best_x.copy(), max_evals=de_evals)
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()

        # Phase 3: (1+1)-ES polish from the best point
        if time_check() and best_f < np.inf:
            remaining = budget - calls_made
            if remaining > 20:
                es_evals = min(60, remaining)
                bx, bf = es_polish(best_x.copy(), max_evals=es_evals)
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()

        # Phase 4: If budget remains, try a few more Nelder-Mead from diverse points
        if time_check() and best_f < np.inf:
            remaining = budget - calls_made
            if remaining > 30:
                for _ in range(5):
                    if not time_check():
                        break
                    if calls_made >= budget:
                        break
                    sp = rng.uniform(-3.0, 3.0, dim)
                    bx, bf = nelder_mead(sp.copy(), max_evals=min(20, budget - calls_made))

        # Phase 5: Final (1+1)-ES polish
        if time_check() and best_f < np.inf:
            remaining = budget - calls_made
            if remaining > 10:
                bx, bf = es_polish(best_x.copy(), max_evals=min(30, remaining))
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
