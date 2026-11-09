"""CMA-ES with Nelder-Mead polish for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with Nelder-Mead polish for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin vs 300s/330s kill

        def timed_out():
            return time.time() > deadline

        rng = np.random.default_rng(20240611)

        best_x = np.zeros(dim)
        best_f = float('inf')
        n_calls = 0

        def call_f(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget or timed_out():
                return None
            y = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(y)
            n_calls += 1
            if f_val < best_f:
                best_f = float(f_val)
                best_x = y.copy()
            return f_val

        def diagonal_cma_es(x0, max_calls):
            """
            Diagonal CMA-ES: adapts per-dimension step sizes.
            """
            nonlocal n_calls, best_x, best_f

            # Population size
            lam = 4 * int(np.log(dim)) + 4  # ~20 for dim=5
            mu = lam // 2
            c_sigma = (mu_eff + 2) / (dim + 2) ** 1.5 + 2 / (dim + 2)
            c_c = (4 + mu_eff / dim) / (dim + 4)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1) - 1) + c_sigma)
            c_sigma = (mu_eff + 2) / (dim + 2) ** 1.5 + 2 / (dim + 2)
            c_c = (4 + mu_eff / dim) / (dim + 4)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1) - 1) + c_sigma)
            mu_eff = mu

            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Initial step size
            sigma = 1.0
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            diag_sigma = np.ones(dim)  # Per-dimension step sizes

            mean = x0.copy()
            calls_used = 0

            while calls_used < max_calls and n_calls < budget and not timed_out():
                # Generate population
                pop = np.zeros((lam, dim))
                fvals = np.zeros(lam)

                for i in range(lam):
                    if n_calls >= budget or timed_out():
                        return
                    z = rng.standard_normal(dim)
                    # Use diagonal adaptation
                    x_candidate = mean + sigma * diag_sigma * z
                    x_candidate = np.clip(x_candidate, -5.0, 5.0)
                    f = call_f(x_candidate)
                    if f is None:
                        return
                    pop[i] = x_candidate
                    fvals[i] = f
                    calls_used += 1

                # Sort by fitness
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                # Update mean (weighted average of top mu)
                new_mean = np.sum(weights[:, None] * pop[:mu], axis=0)

                # Update diagonal step sizes based on progress
                # Measure improvement in each dimension
                improvement = new_mean - mean
                for d in range(dim):
                    if abs(improvement[d]) > 1e-10:
                        # Adapt step size based on how well we moved
                        target = abs(improvement[d]) / (sigma * max(abs(z[d]), 1e-10)) if sigma > 0 else 1.0
                        diag_sigma[d] = np.clip(diag_sigma[d] * (1.0 + 0.5 * np.tanh(target - 1.0)), 0.01, 10.0)
                    else:
                        # No progress, shrink slightly
                        diag_sigma[d] = max(diag_sigma[d] * 0.95, 0.01)

                # Update global sigma
                sigma = max(sigma * 0.99, 0.001)

                # Check convergence
                if np.std(fvals[:mu]) < 1e-10:
                    break

                mean = new_mean

                # Safety: if we're stuck, try a larger step
                if calls_used % 50 == 0 and np.std(fvals[:mu]) > 1e-8:
                    sigma *= 1.5

            return mean, fvals[0]

        def nelder_mead(x0, max_calls, init_step=0.1):
            """
            Nelder-Mead implementation with call budget tracking.
            Returns early if budget or time is exhausted.
            """
            nonlocal n_calls, best_x, best_f

            # Initialize simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0.copy()
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += init_step

            # Evaluate initial simplex
            fvals = np.zeros(dim + 1)
            for i in range(dim + 1):
                f = call_f(simplex[i])
                if f is None:
                    return
                fvals[i] = f

            # Sort simplex by fvals
            def sort_simplex():
                nonlocal simplex, fvals
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

            sort_simplex()

            calls_used = dim + 1
            calls_limit = max_calls

            for _ in range(max_calls):
                if n_calls >= budget or timed_out():
                    break

                # Check convergence
                if np.std(fvals) < 1e-8:
                    break

                # Calculate centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = call_f(x_r)
                if f_r is None:
                    break
                calls_used += 1

                # Sort to see where x_r fits
                if fvals[0] <= f_r < fvals[-1]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                    sort_simplex()
                elif f_r < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = call_f(x_e)
                    if f_e is None:
                        break
                    calls_used += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                    sort_simplex()
                else:
                    # Contraction
                    rho = 0.5
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = call_f(x_c)
                    if f_c is None:
                        break
                    calls_used += 1
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                        sort_simplex()
                    else:
                        # Shrink
                        sigma_shrink = 0.5
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            f = call_f(simplex[i])
                            if f is None:
                                return
                            fvals[i] = f
                            calls_used += 1
                        sort_simplex()

                if calls_used >= calls_limit:
                    break

            return simplex[0], fvals[0]

        # Phase 1: Diagonal CMA-ES for global exploration
        x_start = np.zeros(dim)
        cma_mean, cma_f = diagonal_cma_es(x_start, max_calls=300)

        # Phase 2: Nelder-Mead polish from CMA-ES result
        if n_calls < budget and not timed_out():
            remaining = budget - n_calls
            nm_max = min(remaining, 150)
            nm_x, nm_f = nelder_mead(cma_mean, max_calls=nm_max, init_step=0.05)

        # Phase 3: If still have budget, try a few more CMA-ES restarts from best
        if n_calls < budget and not timed_out():
            for restart in range(3):
                if n_calls >= budget - 20 or timed_out():
                    break
                # Perturb best point
                x_restart = best_x + 0.3 * rng.standard_normal(dim)
                x_restart = np.clip(x_restart, -5.0, 5.0)
                remaining = budget - n_calls - 10
                if remaining < 30:
                    break
                cma_mean2, cma_f2 = diagonal_cma_es(x_restart, max_calls=min(remaining, 80))
                # Polish with NM
                if n_calls < budget and not timed_out():
                    nm_remaining = budget - n_calls
                    if nm_remaining >= 10:
                        nm_x2, nm_f2 = nelder_mead(cma_mean2, max_calls=min(nm_remaining, 30), init_step=0.03)

        return best_x, best_f
# EVOLVE-BLOCK-END
