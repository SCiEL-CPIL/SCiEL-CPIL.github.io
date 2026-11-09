"""Multi-start CMA-ES with diverse initializations for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with adaptive restarts and diverse initializations."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget or check_time():
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                f = float(func(x_clipped))
                return f, x_clipped
            except Exception:
                return None, None

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def cma_es(x0, sigma0, max_calls, rng):
            """
            Runs CMA-ES.
            Returns (best_x, best_f, calls_used, stagnation_detected).
            """
            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2

            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            local_best_x = x.copy()
            local_best_f = np.inf
            calls_used = 0
            stagnation_counter = 0
            no_improve_threshold = 4 * lam

            # Initial population
            pop = np.zeros((lam, n))
            fvals = np.zeros(lam)
            for i in range(lam):
                if calls_used >= max_calls or check_time():
                    return local_best_x, local_best_f, calls_used, True
                z = rng.standard_normal(n)
                x_trial = x + sigma * z
                f, x_eval = call_func(x_trial)
                if f is None:
                    return local_best_x, local_best_f, calls_used, True
                calls_used += 1
                pop[i] = x_eval
                fvals[i] = f
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = x_eval.copy()

            # Main loop
            while calls_used < max_calls and not check_time():
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                x_new = np.sum(weights[:, None] * pop[:mu], axis=0)

                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    Cinv_sqrt = np.eye(n)

                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * Cinv_sqrt @ (x_new - x)
                x = x_new

                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (calls_used // lam)))) < (1.4 + 2.0 / (n + 1)) * chi_n
                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * (x_new - x)

                C = (1.0 + c1 * hsig * (1.0 - hsig) - c1 - cmu) * C + c1 * np.outer(pc, pc)
                for i in range(mu):
                    y_i = (pop[i] - x) / sigma
                    C += cmu * weights[i] * np.outer(y_i, y_i)
                C = (C + C.T) / 2.0

                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-15, 1e6)

                improved_this_gen = False
                for i in range(lam):
                    if calls_used >= max_calls or check_time():
                        break
                    z = rng.standard_normal(n)
                    x_trial = x + sigma * (Cinv_sqrt @ z)
                    f, x_eval = call_func(x_trial)
                    if f is None:
                        return local_best_x, local_best_f, calls_used, True
                    calls_used += 1
                    pop[i] = x_eval
                    fvals[i] = f
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x_eval.copy()
                        improved_this_gen = True

                if improved_this_gen:
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                if stagnation_counter > no_improve_threshold:
                    return local_best_x, local_best_f, calls_used, True

                if sigma < 1e-10:
                    return local_best_x, local_best_f, calls_used, True

            return local_best_x, local_best_f, calls_used, True

        rng = np.random.default_rng(42)

        # Phase 1: Diverse initial sampling with structured spread
        # Use Latin-hypercube-inspired points across the domain
        init_points = []
        # Origin
        init_points.append(np.zeros(dim))
        # Structured spread: sample at different radii and angles
        n_structured = 15
        for i in range(n_structured):
            # Vary the spread from small to large
            spread = 0.5 + 3.5 * (i / max(1, n_structured - 1))
            pt = rng.uniform(-spread, spread, dim)
            init_points.append(pt)
        # A few fully random points
        for _ in range(5):
            init_points.append(rng.uniform(-5, 5, dim))

        # Evaluate all initial points
        for x in init_points:
            if check_time() or n_calls >= budget:
                break
            f, x_eval = call_func(x)
            if f is not None:
                update_best(x_eval, f)

        if n_calls >= budget or check_time():
            return best_x, best_f

        # Phase 2: Multi-start CMA-ES
        # Select top diverse starting points from initial sampling
        # Sort initial points by fitness, pick top 3 distinct ones
        # Re-evaluate to get fitness values
        init_fitness = []
        for x in init_points:
            # We already evaluated these, but we need to track which had best f
            # Actually we should have tracked this. Let's just use best_x and a few others.
            pass

        # Simpler: use best_x, and two other diverse points
        # Find second and third best from initial points by re-checking
        # Actually, let's just use best_x and random diverse points for restarts
        remaining = budget - n_calls
        if remaining < 15:
            return best_x, best_f

        # Number of CMA-ES restarts: 3
        n_restarts = 3
        calls_per_restart = max(remaining // n_restarts, 20)

        # Starting points for each CMA-ES run
        starts = [best_x.copy()]
        # Second start: a point offset from best
        starts.append(best_x + rng.uniform(-1.5, 1.5, dim))
        # Third start: a random point in a moderate region
        starts.append(rng.uniform(-3, 3, dim))

        for i in range(n_restarts):
            if n_calls >= budget or check_time():
                break
            alloc = calls_per_restart
            if i == n_restarts - 1:
                alloc = budget - n_calls  # Use all remaining for last run

            res_x, res_f, used, stagnated = cma_es(starts[i], 1.0, alloc, rng)

            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()

            # If not stagnated, we made good progress; consider stopping early
            # if we're very close to convergence
            if not stagnated and sigma if 'sigma' in dir() else True:
                pass

        # Phase 3: Local Polish with remaining budget
        if n_calls < budget and not check_time():
            local_sigma = 0.1
            for _ in range(budget - n_calls):
                if check_time() or n_calls >= budget:
                    break
                x_trial = best_x + local_sigma * rng.standard_normal(dim)
                f, x_eval = call_func(x_trial)
                if f is not None:
                    if f < best_f:
                        best_f = f
                        best_x = x_eval.copy()
                        local_sigma *= 0.8
                    else:
                        local_sigma *= 0.95
                        if local_sigma < 1e-4:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
