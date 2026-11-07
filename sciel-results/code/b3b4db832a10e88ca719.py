"""Optimizer for BBOB-style functions using adaptive Differential Evolution with Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Population-based optimizer using adaptive Differential Evolution (DE/rand/1)
    with Nelder-Mead final polish. Designed for 5D BBOB functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s
        dim = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            val = float(func(x))
            if not np.isfinite(val):
                return 1e10
            return val

        rng = np.random.default_rng(123)
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = np.array(x, dtype=float)
                best_f = f

        def check_time():
            return time.time() < deadline and calls[0] < budget

        # Phase 1: Initial diverse sampling
        n_initial = min(20, budget)
        points = [np.zeros(dim)]
        # Add some structured points
        for i in range(dim):
            for val in [-2.0, 2.0]:
                p = np.zeros(dim)
                p[i] = val
                points.append(p)
        # Fill with random points
        while len(points) < n_initial:
            points.append(rng.uniform(-5, 5, dim))

        for x in points:
            if not check_time():
                break
            x = np.clip(x, -5, 5)
            try:
                f = wrapped(x)
                update_best(x, f)
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)

        # Phase 2: Adaptive Differential Evolution
        def adaptive_de():
            nonlocal calls, best_x, best_f, deadline
            n = dim
            pop_size = 15  # reasonable for 5D with 300 budget
            max_gen = 25
            F = 0.8  # mutation factor
            CR = 0.9  # crossover rate

            # Initialize population around best point with diversity
            pop = [best_x.copy()]
            for _ in range(pop_size - 1):
                x = best_x + rng.standard_normal(n) * 2.0
                x = np.clip(x, -5, 5)
                pop.append(x)

            pop_f = [np.inf] * pop_size
            for i, x in enumerate(pop):
                if not check_time():
                    return
                try:
                    pop_f[i] = wrapped(x)
                except Exception:
                    pop_f[i] = 1e10
                update_best(x, pop_f[i])

            # Sort population by fitness
            idx = np.argsort(pop_f)
            pop = [pop[i] for i in idx]
            pop_f = [pop_f[i] for i in idx]

            no_improve_gen = 0
            F_current = F

            for gen in range(max_gen):
                if not check_time():
                    break

                new_pop = []
                new_pop_f = []
                gen_improved = False

                for i in range(pop_size):
                    # Adaptive mutation factor
                    if gen % 5 == 0 and gen > 0:
                        # Adjust F based on recent performance
                        if no_improve_gen > 3:
                            F_current = min(1.2, F_current * 1.1)  # increase exploration
                        else:
                            F_current = max(0.5, F_current * 0.95)  # decrease for exploitation

                    # DE/rand/1: pick 3 distinct random individuals
                    candidates = list(range(pop_size))
                    candidates.remove(i)
                    a, b, c = rng.choice(candidates, 3, replace=False)

                    mutant = pop[a] + F_current * (pop[b] - pop[c])
                    mutant = np.clip(mutant, -5, 5)

                    # Binary crossover
                    trial = mutant.copy()
                    for j in range(n):
                        if rng.random() < CR or j == rng.randint(n):
                            trial[j] = mutant[j]
                        else:
                            trial[j] = pop[i][j]

                    try:
                        f_trial = wrapped(trial)
                    except Exception:
                        f_trial = 1e10

                    if f_trial <= pop_f[i]:
                        new_pop.append(trial)
                        new_pop_f.append(f_trial)
                        if f_trial < best_f:
                            gen_improved = True
                            update_best(trial, f_trial)
                    else:
                        new_pop.append(pop[i])
                        new_pop_f.append(pop_f[i])

                # Sort new population
                idx = np.argsort(new_pop_f)
                pop = [new_pop[i] for i in idx]
                pop_f = [new_pop_f[i] for i in idx]

                if gen_improved:
                    no_improve_gen = 0
                else:
                    no_improve_gen += 1

                # Restart if stuck for too long
                if no_improve_gen > 8:
                    # Re-initialize some population members
                    for i in range(pop_size // 2):
                        if i != 0:  # keep best
                            pop[i] = best_x + rng.standard_normal(n) * 1.5
                            pop[i] = np.clip(pop[i], -5, 5)
                            try:
                                pop_f[i] = wrapped(pop[i])
                            except Exception:
                                pop_f[i] = 1e10
                            update_best(pop[i], pop_f[i])
                    no_improve_gen = 0
                    F_current = 0.8  # reset mutation factor

                # Early termination if very close to convergence
                if gen > 5 and abs(pop_f[0] - best_f) < 1e-12 and no_improve_gen > 5:
                    break

        adaptive_de()

        # Phase 3: Nelder-Mead polish from best point
        if best_x is not None and check_time() and calls[0] < budget - 10:
            def nm_obj(x):
                if not check_time():
                    return 1e10
                x = np.clip(x, -5, 5)
                try:
                    return wrapped(x)
                except Exception:
                    return 1e10

            try:
                res = minimize(nm_obj, best_x, method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12,
                                        'adaptive': True})
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # Phase 4: Final fine-tuning with small perturbations
        if best_x is not None and check_time() and calls[0] < budget - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            no_improve = 0
            max_no_improve = 20

            while check_time() and calls[0] < budget - 2:
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                try:
                    fy = wrapped(y)
                except Exception:
                    break

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.05
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1

                if no_improve > max_no_improve or sigma < 1e-12:
                    break

        # Fallback
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
