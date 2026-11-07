"""Optimizer for BBOB-style functions using DE with adaptive F and NM polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Population-based Differential Evolution (DE/rand/1) with adaptive mutation factor,
    followed by Nelder-Mead polishing and a final (1+1)-ES refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        dim = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            return float(func(x))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: Initial diverse sampling
        n_init = 15
        for i in range(n_init):
            if calls[0] >= budget or time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            elif i == 1:
                x = np.full(dim, 1.0)
            elif i == 2:
                x = np.full(dim, -1.0)
            else:
                x = rng.uniform(-5, 5, dim)
            try:
                f_val = wrapped(x)
                update_best(x, f_val)
            except Exception:
                pass

        # Phase 2: Multi-start L-BFGS-B for smooth functions
        if best_x is not None and calls[0] < budget:
            starts = [best_x.copy()]
            # Add perturbed starts around best
            for _ in range(5):
                s = best_x + rng.standard_normal(dim) * 0.5
                s = np.clip(s, -5, 5)
                starts.append(s)
            # Add some random starts
            for _ in range(3):
                s = rng.uniform(-5, 5, dim)
                starts.append(s)

            for s in starts:
                if calls[0] >= budget or time.time() > deadline:
                    break
                try:
                    res = minimize(
                        wrapped,
                        s,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'maxls': 15, 'gtol': 1e-4, 'ftol': 1e-8}
                    )
                    update_best(res.x, res.fun)
                except Exception:
                    pass

        # Phase 3: Differential Evolution with adaptive F
        if best_x is not None and calls[0] < budget:
            pop_size = 12
            remaining = budget - calls[0]
            # Each generation: pop_size evaluations for mutants + pop_size for crossovers = 2*pop_size
            # But we can be more efficient: only evaluate mutant if it's promising
            n_gen = max(1, min(15, remaining // (pop_size * 2)))

            # Initialize population
            pop = [best_x.copy()]
            for _ in range(pop_size - 1):
                if rng.random() < 0.4:
                    p = best_x + rng.standard_normal(dim) * 1.5
                else:
                    p = rng.uniform(-5, 5, dim)
                pop.append(np.clip(p, -5, 5))

            # Evaluate initial population
            pop_f = []
            for p in pop:
                if calls[0] >= budget or time.time() > deadline:
                    break
                try:
                    f_val = wrapped(p)
                    pop_f.append(f_val)
                    update_best(p, f_val)
                except Exception:
                    break

            if len(pop_f) < 3:
                return best_x, best_f

            pop = pop[:len(pop_f)]
            pop_f = pop_f[:len(pop_f)]
            pop_size = len(pop)

            # Adaptive DE loop
            F = 0.8
            CR = 0.5
            for gen in range(n_gen):
                if calls[0] >= budget or time.time() > deadline:
                    break

                new_pop = []
                new_pop_f = []

                for i in range(pop_size):
                    if calls[0] >= budget or time.time() > deadline:
                        break

                    # Select distinct indices
                    other_indices = [j for j in range(pop_size) if j != i]
                    if len(other_indices) < 2:
                        break
                    r1, r2, r3 = rng.choice(other_indices, 3, replace=False)

                    # Mutation: DE/rand/1
                    v = pop[r1] + F * (pop[r2] - pop[r3])
                    v = np.clip(v, -5, 5)

                    # Crossover
                    u = np.zeros(dim)
                    j_rand = rng.integers(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            u[j] = v[j]
                        else:
                            u[j] = pop[i][j]

                    # Evaluate candidate
                    try:
                        f_u = wrapped(u)
                    except Exception:
                        continue

                    # Selection
                    if f_u <= pop_f[i]:
                        new_pop.append(u)
                        new_pop_f.append(f_u)
                        update_best(u, f_u)
                    else:
                        new_pop.append(pop[i])
                        new_pop_f.append(pop_f[i])

                if len(new_pop) < pop_size:
                    break

                pop = new_pop
                pop_f = new_pop_f

                # Adapt F based on success
                # If many new points were accepted, keep F; otherwise increase exploration
                accepted = sum(1 for i in range(pop_size) if not np.allclose(pop[i], new_pop[i]))
                if accepted > pop_size * 0.5:
                    F = max(0.4, F * 0.9)
                else:
                    F = min(1.2, F * 1.1)

                # Also adapt CR
                if accepted > pop_size * 0.3:
                    CR = max(0.3, CR * 0.95)
                else:
                    CR = min(0.7, CR * 1.05)

        # Phase 4: Nelder-Mead polish
        if best_x is not None and calls[0] < budget:
            try:
                remaining = budget - calls[0]
                nm_maxiter = min(100, max(10, remaining // 2))
                res_nm = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': nm_maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                update_best(res_nm.x, res_nm.fun)
            except Exception:
                pass

        # Phase 5: (1+1)-ES fine-tuning
        if best_x is not None and calls[0] < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.02
            no_improve = 0
            max_no_improve = 20

            while calls[0] < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    fy = wrapped(y)
                except Exception:
                    break

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.03
                    no_improve = 0
                else:
                    sigma *= 0.92
                    no_improve += 1

                if no_improve > max_no_improve or sigma < 1e-8:
                    break

        # Phase 6: Final L-BFGS-B polish
        if best_x is not None and calls[0] < budget - 3:
            try:
                res_final = minimize(
                    wrapped,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 30, 'maxls': 10, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                update_best(res_final.x, res_final.fun)
            except Exception:
                pass

        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            best_f = wrapped(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
