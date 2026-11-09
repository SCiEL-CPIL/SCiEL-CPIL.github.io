"""CMA-ES with adaptive restarts and local polish for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and local polish for multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s kill

        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget:
                return None
            n_calls += 1
            try:
                x_arr = np.asarray(x, dtype=np.float64)
                x_clipped = np.clip(x_arr, -5.0, 5.0)
                return float(func(x_clipped)), x_clipped
            except Exception:
                return None, None

        def update(x, f):
            nonlocal best_x, best_f
            if f is not None and f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=np.float64).copy()

        def local_polish(x0, max_calls, rng):
            """Simple coordinate-descent + random jitter polish."""
            x = np.asarray(x0, dtype=np.float64).copy()
            res = call_func(x)
            if res[0] is None:
                return x, np.inf
            f, x_eval = res
            best_local_f = f
            best_local_x = x_eval.copy()
            step = 0.5
            for _ in range(max_calls):
                if check_time() or n_calls >= budget:
                    break
                improved = False
                for d in range(dim):
                    if check_time() or n_calls >= budget:
                        break
                    for sign in (1.0, -1.0):
                        if n_calls >= budget or check_time():
                            break
                        trial = x.copy()
                        trial[d] += sign * step
                        trial = np.clip(trial, -5.0, 5.0)
                        res = call_func(trial)
                        if res[0] is None:
                            continue
                        ft, x_trial = res
                        if ft < best_local_f - 1e-12:
                            x = x_trial.copy()
                            best_local_f = ft
                            best_local_x = x.copy()
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-5:
                        # Try random jitter
                        for _ in range(3):
                            if n_calls >= budget or check_time():
                                break
                            trial = x + rng.normal(0, 0.1, dim)
                            trial = np.clip(trial, -5.0, 5.0)
                            res = call_func(trial)
                            if res[0] is None:
                                continue
                            ft, x_trial = res
                            if ft < best_local_f - 1e-12:
                                x = x_trial.copy()
                                best_local_f = ft
                                best_local_x = x.copy()
                                improved = True
                                break
                        if not improved:
                            break
                    else:
                        # Occasionally add a larger jump to escape local minima
                        if rng.random() < 0.1:
                            trial = x + rng.normal(0, step * 2, dim)
                            trial = np.clip(trial, -5.0, 5.0)
                            res = call_func(trial)
                            if res[0] is None:
                                continue
                            ft, x_trial = res
                            if ft < best_local_f - 1e-12:
                                x = x_trial.copy()
                                best_local_f = ft
                                best_local_x = x.copy()
            return best_local_x, best_local_f

        def cma_es(x0, sigma0, max_calls, rng):
            """Correct CMA-ES with call tracking."""
            n = dim
            lam = max(6, 4 + int(3 * np.log(n)))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = (2.0 / ((n + 1.3) ** 2 + mueff)) * (1.0 - cc)
            cmu = (2.0 * (mueff / n - 1.0) / ((n + 2) ** 2 + mueff)) * (
                min(1.0, c1) - c1
            )

            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            sigma = sigma0
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

            x = np.asarray(x0, dtype=np.float64).copy()
            calls_used = 0
            local_best_x = x.copy()
            local_best_f = np.inf
            gen = 0
            stagnation = 0

            def sample_population(mean, sig, C):
                """Generate population using eigendecomposition of C."""
                pop = np.zeros((lam, n))
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    Cinv_sqrt = np.eye(n)
                for i in range(lam):
                    z = rng.standard_normal(n)
                    pop[i] = mean + sig * (Cinv_sqrt @ z)
                return pop, Cinv_sqrt

            # Initial population
            pop, Cinv_sqrt = sample_population(x, sigma, C)
            fvals = np.zeros(lam)
            for i in range(lam):
                if calls_used >= max_calls or check_time():
                    break
                res = call_func(pop[i])
                if res[0] is None:
                    return local_best_x, local_best_f, calls_used
                f, x_eval = res
                calls_used += 1
                fvals[i] = f
                pop[i] = x_eval
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = x_eval.copy()

            # Main loop
            while calls_used < max_calls and not check_time():
                # Sort by fitness
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                # Check stagnation
                if fvals[0] < local_best_f - 1e-10:
                    stagnation = 0
                else:
                    stagnation += 1
                if stagnation > 5 and gen > 3:
                    break

                # Update mean
                x_new = np.zeros(n)
                for i in range(mu):
                    x_new += weights[i] * pop[i]
                x = x_new

                # Update ps and pc BEFORE updating C (use old Cinv_sqrt)
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * Cinv_sqrt @ (x - x_new)
                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (gen + 1)))) < (1.4 + 2.0 / (n + 1)) * chi_n
                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * Cinv_sqrt @ (x - x_new)

                # Update covariance (correct formula)
                C = (1.0 - c1 - cmu) * C
                C += c1 * (np.outer(pc, pc) + (1.0 + (1.0 - cc) * cc) * C - C)
                for i in range(mu):
                    diff = pop[i] - x
                    C += cmu * weights[i] * np.outer(diff, diff)
                C = (C + C.T) / 2.0

                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-15, 1e6)

                # Check convergence
                if sigma < 1e-10:
                    break

                # Generate new population
                pop, Cinv_sqrt = sample_population(x, sigma, C)
                for i in range(lam):
                    if calls_used >= max_calls or check_time():
                        break
                    res = call_func(pop[i])
                    if res[0] is None:
                        return local_best_x, local_best_f, calls_used
                    f, x_eval = res
                    calls_used += 1
                    fvals[i] = f
                    pop[i] = x_eval
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x_eval.copy()

                gen += 1

            return local_best_x, local_best_f, calls_used

        rng = np.random.default_rng(12345)

        # Phase 1: Initial stratified sampling for fast early progress
        n_initial = min(20, max(10, budget // 25))
        # Use a quasi-random (Halton-like) spread for better coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        for i in range(n_initial):
            if check_time() or n_calls >= budget:
                break
            x = np.array([halton(i + 1, b) for b in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]])
            x = x * 10.0 - 5.0  # map to [-5, 5]
            res = call_func(x)
            if res[0] is not None:
                update(res[1], res[0])

        # Add a few random points
        for i in range(5):
            if check_time() or n_calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            res = call_func(x)
            if res[0] is not None:
                update(res[1], res[0])

        if best_x is None:
            x = np.zeros(dim)
            res = call_func(x)
            if res[0] is not None:
                update(res[1], res[0])
            if best_x is None:
                return None

        # Phase 2: CMA-ES with adaptive restarts
        remaining = budget - n_calls
        if remaining < 20:
            # Not enough for CMA-ES, do local polish
            px, pf = local_polish(best_x, remaining, rng)
            update(px, pf)
            return best_x, best_f

        # Adaptive restart strategy: run CMA-ES, if it stagnates, restart from best
        # with a fresh covariance and slightly perturbed start
        n_restarts_max = 4
        restarts_done = 0
        cma_budget_min = 30  # minimum calls for a meaningful CMA-ES run

        while remaining >= cma_budget_min and not check_time() and restarts_done < n_restarts_max:
            # Allocate budget: give more to earlier restarts
            if restarts_done == 0:
                max_calls = min(remaining - 3 * cma_budget_min, budget - n_calls)
                max_calls = max(max_calls, cma_budget_min)
            else:
                max_calls = min(remaining, budget - n_calls)
                max_calls = max(max_calls, cma_budget_min)

            if max_calls < cma_budget_min:
                break

            # Start point: best so far with small perturbation for restarts > 0
            if restarts_done == 0:
                x0 = best_x.copy()
                sigma0 = 1.5
            else:
                x0 = best_x + rng.normal(0, 0.5, dim)
                x0 = np.clip(x0, -5.0, 5.0)
                sigma0 = 1.0

            res_x, res_f, used = cma_es(x0, sigma0, max_calls, rng)
            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()
            remaining = budget - n_calls
            restarts_done += 1

            # Check if we should continue: if no improvement, try a different strategy
            if restarts_done > 0 and res_f >= best_f - 1e-8:
                # No improvement, try a random restart
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 2.0
                if remaining >= cma_budget_min and not check_time():
                    res_x, res_f, used = cma_es(x0, sigma0, min(remaining, budget - n_calls), rng)
                    if res_f < best_f:
                        best_f = res_f
                        best_x = res_x.copy()
                    remaining = budget - n_calls

        # Phase 3: Local polish with remaining budget
        if remaining > 5 and not check_time():
            px, pf = local_polish(best_x, min(remaining, 50), rng)
            update(px, pf)

        return best_x, best_f
# EVOLVE-BLOCK-END
