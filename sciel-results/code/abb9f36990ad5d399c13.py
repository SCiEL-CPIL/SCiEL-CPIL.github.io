import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Track evaluations
        eval_count = [0]

        def safe_func(x):
            try:
                x_arr = np.asarray(x, dtype=float)
                if x_arr.shape != (dim,):
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

        rng = np.random.default_rng(12345)

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_left():
            return timeout - (time.time() - start_time)

        def evals_left():
            return max(0, budget - eval_count[0])

        # Phase 1: Differential Evolution for global exploration
        # DE is robust for multimodal functions like Rastrigin and Gallagher
        n_pop = 30  # population size
        F = 0.8     # differential weight
        CR = 0.9    # crossover rate

        # Initialize population
        pop = rng.uniform(-5, 5, (n_pop, dim))
        pop_f = np.array([safe_func(x) for x in pop])

        best_idx = np.argmin(pop_f)
        update_best(pop[best_idx], pop_f[best_idx])

        # Run DE for a portion of the budget
        de_evals_target = 150  # Use ~150 evals for DE

        while evals_left() > de_evals_target and time_left() > 50:
            # Select best individual
            best_idx = np.argmin(pop_f)
            best_x_pop = pop[best_idx]
            best_f_pop = pop_f[best_idx]

            # Generate new candidate
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
                update_best(trial, trial_f)

        # Phase 2: Local refinement using L-BFGS-B from multiple starting points
        # Use the best few points from DE as starting points
        # Sort population by fitness
        sorted_indices = np.argsort(pop_f)
        n_local_starts = min(5, n_pop)
        local_starts = pop[sorted_indices[:n_local_starts]]

        # Also add a few random points for diversity
        if evals_left() > 20:
            extra_starts = rng.uniform(-5, 5, (3, dim))
            local_starts = np.vstack([local_starts, extra_starts])

        for x0 in local_starts:
            if evals_left() < 5 or time_left() < 10:
                break

            # Run L-BFGS-B
            try:
                res = scipy.optimize.minimize(
                    safe_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 30, 'maxfun': 30, 'gtol': 1e-6}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Final polish with Nelder-Mead from the best point
        if evals_left() > 5 and time_left() > 5:
            try:
                res = scipy.optimize.minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 30, 'maxfev': 30, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Ensure we stay within budget
        # If we're close to the eval limit, stop
        return best_x, best_f
