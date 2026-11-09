"""Differential Evolution with adaptive parameters for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE with adaptive F and CR (jDE) for multimodal Weierstrass optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget:
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                return float(func(x_clipped)), x_clipped
            except Exception:
                return None, None

        def update(x, f):
            nonlocal best_x, best_f
            if f is not None and f < best_f:
                best_f = f
                best_x = x.copy()

        def differential_evolution(x0, max_calls, rng):
            """Run jDE with call tracking. Returns (best_x, best_f, calls_used)."""
            n = dim
            # Population size: large for global diversity
            lam = min(20, max(8, n * 4))
            
            # Adaptive parameters (jDE)
            F_base = 0.5
            CR_base = 0.7
            sigma_F = 0.1
            sigma_CR = 0.1
            
            calls_used = 0
            local_best_x = x0.copy()
            local_best_f = np.inf

            # Initialize population
            pop = np.zeros((lam, n))
            fvals = np.zeros(lam)
            
            # First individual is x0
            pop[0] = x0
            res = call_func(pop[0])
            if res[0] is None:
                return local_best_x, local_best_f, calls_used
            fvals[0] = res[0]
            calls_used += 1
            if fvals[0] < local_best_f:
                local_best_f = fvals[0]
                local_best_x = pop[0].copy()

            # Rest of population: random in [-5, 5]
            for i in range(1, lam):
                if check_time() or calls_used >= max_calls:
                    break
                pop[i] = rng.uniform(-5, 5, n)
                res = call_func(pop[i])
                if res[0] is None:
                    return local_best_x, local_best_f, calls_used
                fvals[i] = res[0]
                calls_used += 1
                if fvals[i] < local_best_f:
                    local_best_f = fvals[i]
                    local_best_x = pop[i].copy()

            # Track best indices for mutation
            best_idx = 0
            for i in range(lam):
                if fvals[i] < fvals[best_idx]:
                    best_idx = i

            generation = 0
            while calls_used < max_calls and not check_time():
                generation += 1
                # Sort population
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]
                best_idx = 0

                # Adaptive parameters based on generation progress
                progress = min(1.0, generation / 50.0)
                F_current = F_base + (0.3 - F_base) * progress  # decrease F over time
                CR_current = CR_base + (0.9 - CR_base) * progress  # increase CR over time

                new_pop = pop.copy()
                new_fvals = fvals.copy()

                for i in range(lam):
                    if calls_used >= max_calls or check_time():
                        break

                    # jDE: sample F and CR from normal distributions
                    F_sample = max(0.1, min(1.0, F_current + rng.normal(0, sigma_F)))
                    CR_sample = max(0.1, min(1.0, CR_current + rng.normal(0, sigma_CR)))

                    # DE/rand/1 mutation: select 3 distinct random individuals
                    r1, r2, r3 = rng.choice(lam, size=3, replace=False)
                    while r1 == i:
                        r1 = rng.choice(lam)
                    while r2 == i or r2 == r1:
                        r2 = rng.choice(lam)
                    while r3 == i or r3 == r1 or r3 == r2:
                        r3 = rng.choice(lam)

                    # Mutation
                    v = pop[r1] + F_sample * (pop[r2] - pop[r3])

                    # Crossover
                    jrand = rng.integers(n)
                    trial = np.zeros(n)
                    for j in range(n):
                        if rng.random() <= CR_sample or j == jrand:
                            trial[j] = v[j]
                        else:
                            trial[j] = pop[i][j]

                    # Clip to bounds
                    trial = np.clip(trial, -5, 5)

                    res = call_func(trial)
                    if res[0] is None:
                        return local_best_x, local_best_f, calls_used
                    f_trial = res[0]
                    calls_used += 1

                    # Selection
                    if f_trial <= fvals[i]:
                        new_pop[i] = trial
                        new_fvals[i] = f_trial
                        if f_trial < local_best_f:
                            local_best_f = f_trial
                            local_best_x = trial.copy()

                pop = new_pop
                fvals = new_fvals

                # Convergence check: if std is very small, break
                if np.std(fvals) < 1e-10:
                    break

                # Elite preservation: ensure best individual survives
                best_in_new = np.argmin(fvals)
                if fvals[best_in_new] > local_best_f:
                    pop[0] = local_best_x
                    fvals[0] = local_best_f

            return local_best_x, local_best_f, calls_used

        def local_polish(x_start, max_calls, rng):
            """Simple (1+1)-ES local refinement."""
            n = dim
            x = x_start.copy()
            res = call_func(x)
            if res[0] is None:
                return x, np.inf, 0
            f = res[0]
            calls_used = 1
            sigma = 0.1  # Start with small step size

            while calls_used < max_calls and not check_time():
                # Generate trial
                z = rng.standard_normal(n)
                trial = x + sigma * z
                trial = np.clip(trial, -5, 5)

                res = call_func(trial)
                if res[0] is None:
                    break
                f_trial = res[0]
                calls_used += 1

                if f_trial < f:
                    x = trial
                    f = f_trial
                    sigma *= 1.5  # Increase step on success
                else:
                    sigma *= 0.7  # Decrease step on failure

                if sigma < 1e-8:
                    break

            return x, f, calls_used

        rng = np.random.default_rng(42)

        # Phase 1: Initial sampling to find promising regions
        n_initial = min(15, budget // 30)
        for i in range(n_initial):
            if check_time() or n_calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            res = call_func(x)
            if res[0] is None:
                break
            f, x_eval = res
            update(x_eval, f)

        if best_x is None:
            x = np.zeros(dim)
            res = call_func(x)
            if res[0] is not None:
                update(res[1], res[0])
            if best_x is None:
                return None

        # Phase 2: Main DE optimization
        remaining = budget - n_calls
        if remaining < 20:
            return best_x, best_f

        # Allocate 70% to DE, 30% to local polish
        de_budget = int(remaining * 0.7)
        polish_budget = remaining - de_budget

        # Run DE from best point found
        res_x, res_f, used = differential_evolution(best_x.copy(), de_budget, rng)
        if res_f < best_f:
            best_f = res_f
            best_x = res_x.copy()

        # Phase 3: Local polish if budget remains
        if polish_budget > 5 and not check_time():
            res_x, res_f, used = local_polish(best_x.copy(), polish_budget, rng)
            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
