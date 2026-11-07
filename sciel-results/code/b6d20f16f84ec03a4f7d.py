import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 285.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Track evaluations
        eval_count = [0]

        def safe_func(x):
            try:
                x_arr = np.asarray(x, dtype=float)
                if x_arr.ndim != 1 or x_arr.shape[0] != dim:
                    x_arr = x_arr.reshape(dim)
                x_clipped = np.clip(x_arr, -5.0, 5.0)
                val = float(func(x_clipped))
                eval_count[0] += 1
                if not np.isfinite(val):
                    return 1e10
                return val
            except Exception:
                eval_count[0] += 1
                return 1e10

        best_x = np.zeros(dim)
        best_f = float('inf')

        rng = np.random.default_rng(42)

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_left():
            return timeout - (time.time() - start_time)

        def evals_left():
            return max(0, budget - eval_count[0])

        # Phase 1: Differential Evolution with adaptive parameters
        # DE is robust for multimodal functions
        n_pop = 25
        F = 0.7
        CR = 0.8

        # Initialize population with diverse points
        pop = rng.uniform(-5, 5, (n_pop, dim))
        # Add a few points near origin and corners for coverage
        pop[0] = np.zeros(dim)
        pop[1] = np.full(dim, -4.0)
        pop[2] = np.full(dim, 4.0)
        pop[3] = rng.uniform(-3, 3, dim)
        pop[4] = rng.uniform(-3, 3, dim)

        pop_f = np.array([safe_func(x) for x in pop])

        best_idx = np.argmin(pop_f)
        update_best(pop[best_idx].copy(), pop_f[best_idx])

        # Run DE for a portion of the budget
        de_evals_target = 120  # Use ~120 evals for DE

        no_improve_count = 0
        max_no_improve = 30  # Reset F if stuck

        while evals_left() > de_evals_target and time_left() > 60:
            # Adaptive F: reduce F if stuck, increase if making progress
            if no_improve_count > max_no_improve:
                F = max(0.4, F - 0.1)
                no_improve_count = 0
            elif no_improve_count > 15:
                F = min(1.0, F + 0.05)

            # Select best individual
            best_idx = np.argmin(pop_f)
            best_x_pop = pop[best_idx].copy()
            best_f_pop = pop_f[best_idx]

            # Generate new candidate using DE strategy
            r1, r2, r3 = rng.choice(n_pop, 3, replace=False)
            v = pop[r1] + F * (pop[r2] - pop[r3])
            v = np.clip(v, -5, 5)

            # Crossover
            trial = np.copy(pop[best_idx])
            j_rand = rng.integers(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]

            # Evaluate
            trial_f = safe_func(trial)

            # Selection
            if trial_f <= pop_f[best_idx]:
                pop[best_idx] = trial
                pop_f[best_idx] = trial_f
                if trial_f < best_f:
                    update_best(trial.copy(), trial_f)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            # Occasionally inject a random point to maintain diversity
            if evals_left() > de_evals_target + 20 and rng.random() < 0.05:
                worst_idx = np.argmax(pop_f)
                new_point = rng.uniform(-5, 5, dim)
                new_f = safe_func(new_point)
                if new_f < pop_f[worst_idx]:
                    pop[worst_idx] = new_point
                    pop_f[worst_idx] = new_f
                    if new_f < best_f:
                        update_best(new_point.copy(), new_f)

        # Phase 2: Local refinement using L-BFGS-B from multiple starting points
        # Sort population by fitness
        sorted_indices = np.argsort(pop_f)
        n_local_starts = min(6, n_pop)
        local_starts = pop[sorted_indices[:n_local_starts]].copy()

        # Also add the global best
        local_starts = np.vstack([local_starts, best_x.reshape(1, -1)])

        for x0 in local_starts:
            if evals_left() < 10 or time_left() < 15:
                break

            # Run L-BFGS-B
            try:
                res = scipy.optimize.minimize(
                    safe_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxfun': 40, 'gtol': 1e-7}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Nelder-Mead polish from the best point
        if evals_left() > 10 and time_left() > 10:
            try:
                res = scipy.optimize.minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'maxfev': 40, 'xatol': 1e-9, 'fatol': 1e-12}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 4: Final coordinate descent polish
        if evals_left() > 5 and time_left() > 5:
            x = best_x.copy()
            f = best_f
            for _ in range(3):
                improved = False
                for d in range(dim):
                    for step in [0.01, 0.001, 0.0001]:
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[d] += direction * step
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = safe_func(x_trial)
                            if f_trial < f:
                                x = x_trial
                                f = f_trial
                                improved = True
                if not improved:
                    break
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

        return best_x, best_f
