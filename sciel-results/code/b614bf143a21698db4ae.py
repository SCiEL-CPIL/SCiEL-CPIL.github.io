"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """
    Optimizer for BBOB f18 (Schaffers F7 ill-conditioned) using DE + local search.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0
        dim = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(7)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        def record(x, f):
            nonlocal best_x, best_f, calls
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def safe_eval(x):
            nonlocal calls
            if calls >= total_budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            try:
                f = float(func(x))
            except Exception:
                calls += 1
                return np.inf
            calls += 1
            return f

        # --- Phase 1: Latin Hypercube Sampling for diverse initial points ---
        n_lhs = min(50, total_budget // 10)
        lhs_points = []
        for _ in range(n_lhs):
            if time.time() > deadline - 3.0 or calls >= total_budget:
                break
            x = rng.permutation(dim).astype(float) * (10.0 / dim) - 5.0
            x = np.clip(x, -5, 5)
            f = safe_eval(x)
            if np.isfinite(f):
                lhs_points.append((f, x.copy()))
            record(x, f if np.isfinite(f) else np.inf)

        # Sort by fitness
        lhs_points.sort(key=lambda t: t[0])

        # --- Phase 2: Differential Evolution ---
        def de_run(budget_for_run, pop, F=0.8, CR=0.9):
            nonlocal calls
            n_pop = len(pop)
            if n_pop < 3:
                return
            # pop is list of (f, x)
            pop = sorted(pop, key=lambda t: t[0])

            # Adaptive F and CR based on progress
            gen = 0
            stall_count = 0
            prev_best = pop[0][0]

            while calls < budget_for_run:
                if time.time() > deadline - 5.0:
                    break

                new_pop = []
                for i in range(n_pop):
                    if calls >= budget_for_run or time.time() > deadline - 3.0:
                        break

                    # DE/rand/1 mutation
                    # Pick 3 distinct indices different from i
                    mask = np.array([j for j in range(n_pop) if j != i])
                    a, b, c = rng.choice(mask, size=3, replace=False)

                    # Adaptive F: start large, shrink as we converge
                    progress = gen / max(1, budget_for_run // n_pop)
                    F_adapt = F * (1.0 - 0.4 * min(1.0, progress))
                    CR_adapt = CR - 0.2 * min(1.0, progress)

                    v = pop[a][1] + F_adapt * (pop[b][1] - pop[c][1])
                    v = np.clip(v, -5, 5)

                    # Crossover
                    j_rand = rng.randint(dim)
                    u = np.where(rng.random(dim) <= CR_adapt, v, pop[i][1])
                    u[j_rand] = v[j_rand]

                    f_u = safe_eval(u)
                    if f_u <= pop[i][0]:
                        new_pop.append((f_u, u.copy()))
                        record(u, f_u)
                    else:
                        new_pop.append(pop[i])

                if len(new_pop) < 3:
                    break

                new_pop.sort(key=lambda t: t[0])
                pop = new_pop

                # Check for improvement
                if pop[0][0] < prev_best - 1e-10:
                    prev_best = pop[0][0]
                    stall_count = 0
                else:
                    stall_count += 1

                gen += 1

                # If stuck too long, inject diversity
                if stall_count > 15 and n_pop > 5:
                    for i in range(min(3, n_pop - 1)):
                        if calls >= budget_for_run:
                            break
                        x_new = rng.uniform(-5, 5, dim)
                        f_new = safe_eval(x_new)
                        record(x_new, f_new)
                        if f_new < pop[-1][0]:
                            pop[-1] = (f_new, x_new.copy())
                    pop.sort(key=lambda t: t[0])
                    stall_count = 0

            return pop

        # Build initial population from LHS + random
        pop = []
        for f, x in lhs_points[:20]:
            pop.append((f, x))

        # Add more random points to fill population
        n_pop_target = 20
        while len(pop) < n_pop_target:
            if time.time() > deadline - 3.0 or calls >= total_budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = safe_eval(x)
            pop.append((f, x))
            record(x, f)

        # Run DE
        de_budget = int(total_budget * 0.55)
        de_run(de_budget, pop)

        # --- Phase 3: Local refinement from best DE point ---
        def nelder_mead(x0, f0, max_iter=30):
            nonlocal calls
            n = dim
            # Simplex
            simplex = [x0.copy()]
            simplexF = [f0]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.5 * (5 - x0[i]) if x0[i] >= 0 else 0.5 * (x0[i] + 5)
                xi = np.clip(xi, -5, 5)
                fi = safe_eval(xi)
                simplex.append(xi)
                simplexF.append(fi)
                record(xi, fi)

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for it in range(max_iter):
                if time.time() > deadline - 3.0 or calls >= total_budget:
                    break

                # Sort
                idx = np.argsort(simplexF)
                simplex = [simplex[i] for i in idx]
                simplexF = [simplexF[i] for i in idx]

                if simplexF[0] < best_f - 1e-10:
                    best_f = simplexF[0]
                    best_x = simplex[0].copy()

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_eval(x_r)
                record(x_r, f_r)

                if f_r < simplexF[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_eval(x_e)
                    record(x_e, f_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplexF[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplexF[-1] = f_r
                elif f_r < simplexF[-2]:
                    simplex[-1] = x_r
                    simplexF[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + 0.5 * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = safe_eval(x_c)
                    record(x_c, f_c)
                    if f_c < simplexF[-1]:
                        simplex[-1] = x_c
                        simplexF[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplexF[i] = safe_eval(simplex[i])
                            record(simplex[i], simplexF[i])
                        if max(abs(np.array(simplex[0]) - np.array(simplex[i]))) < 1e-6:
                            break

            return simplex[0], simplexF[0]

        def coordinate_descent(x0, f0, max_iter=20):
            nonlocal calls
            x = x0.copy()
            f = f0
            step_sizes = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

            for ss in step_sizes:
                if time.time() > deadline - 3.0 or calls >= total_budget:
                    break
                for _ in range(max_iter):
                    improved = False
                    for i in range(dim):
                        if calls >= total_budget or time.time() > deadline - 3.0:
                            break
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[i] += direction * ss
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = safe_eval(x_trial)
                            record(x_trial, f_trial)
                            if f_trial < f - 1e-10:
                                x = x_trial
                                f = f_trial
                                improved = True
                                break
                    if not improved:
                        break

            return x, f

        # Local search from best point
        if calls < total_budget - 30 and time.time() < deadline - 15:
            # Nelder-Mead
            nm_budget = 40
            if time.time() < deadline - 10:
                x_nm, f_nm = nelder_mead(best_x, best_f, max_iter=nm_budget)
                record(x_nm, f_nm)

            # Coordinate descent from NM result and original best
            if calls < total_budget - 20 and time.time() < deadline - 8:
                x_cd, f_cd = coordinate_descent(best_x, best_f, max_iter=15)
                record(x_cd, f_cd)

                if f_nm < best_f - 1e-10:
                    x_cd2, f_cd2 = coordinate_descent(x_nm, f_nm, max_iter=10)
                    record(x_cd2, f_cd2)

        # --- Phase 4: Final fine-tuning with small perturbations ---
        if calls < total_budget - 5 and time.time() < deadline - 5:
            remaining = total_budget - calls
            for scale in [0.05, 0.02, 0.01, 0.005]:
                for _ in range(min(5, remaining)):
                    if time.time() > deadline - 2.0 or calls >= total_budget:
                        break
                    x_try = best_x + scale * rng.standard_normal(dim)
                    x_try = np.clip(x_try, -5, 5)
                    f_try = safe_eval(x_try)
                    record(x_try, f_try)

        return best_x, best_f
# EVOLVE-BLOCK-END
