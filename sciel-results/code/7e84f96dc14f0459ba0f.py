"""Optimizer using CMA-ES with adaptive local polishing for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with adaptive local search polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin from 330s limit
        n = self.dim
        budget = self.budget

        # CMA-ES Parameters tuned for low dimensions (n=5)
        lambda_pop = 10
        mu = lambda_pop // 2
        weights = np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (n + 4)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

        # Initialize
        rng = np.random.default_rng(42)
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        sigma = 1.0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)

        # Local search parameters
        local_search_budget = 30
        stagnation_threshold = 4  # generations without improvement

        def nelder_mead(x0, max_evals, step_scale=1.0):
            """Nelder-Mead local polishing with adaptive step size."""
            simplex = [x0.copy() for _ in range(n + 1)]
            step = max(0.05, 0.3 * sigma * step_scale)
            for i in range(n):
                simplex[i+1] = x0.copy()
                simplex[i+1][i] += step
            
            fvals = np.array([func(s) for s in simplex])
            local_evals = n + 1

            for _ in range(max_evals // (n + 1) + 1):
                if time.time() > deadline - 5:
                    break
                if local_evals >= max_evals:
                    break

                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = fvals[order]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                local_evals += 1

                if fr < fvals[0]:
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    local_evals += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    local_evals += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            local_evals += 1
                        if local_evals >= max_evals:
                            break

            best_local_idx = np.argmin(fvals)
            return simplex[best_local_idx], fvals[best_local_idx], local_evals

        generation = 0
        no_improve = 0
        local_search_count = 0

        while calls < budget:
            if time.time() > deadline - 2:
                break

            # Generate population
            pops = []
            fits = []

            for i in range(lambda_pop):
                if calls >= budget or time.time() > deadline - 2:
                    break
                z = rng.standard_normal(n)
                try:
                    L = np.linalg.cholesky(C)
                    y = np.dot(L, z)
                except:
                    y = z
                x_i = np.clip(x + sigma * y, -5, 5)
                f_i = func(x_i)
                calls += 1
                pops.append(x_i)
                fits.append(f_i)

            if len(fits) == 0:
                break

            # Sort
            sorted_idx = np.argsort(fits)
            pops = [pops[i] for i in sorted_idx]
            fits = [fits[i] for i in sorted_idx]

            # Update best
            if fits[0] < best_fx:
                best_fx = fits[0]
                best_x = pops[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            generation += 1

            # Trigger local search on stagnation
            if no_improve >= stagnation_threshold and calls + local_search_budget < budget:
                if time.time() < deadline - 10:
                    # Adaptive local search budget: more aggressive early, conservative late
                    remaining = budget - calls
                    if remaining > 100:
                        ls_budget = min(local_search_budget, 30)
                    elif remaining > 50:
                        ls_budget = min(local_search_budget, 20)
                    else:
                        ls_budget = min(local_search_budget, 10)
                    
                    local_x, local_f, used_evals = nelder_mead(best_x.copy(), ls_budget)
                    calls += used_evals
                    local_search_count += 1

                    if local_f < best_fx:
                        best_fx = local_f
                        best_x = local_x
                        no_improve = 0
                        # Shrink sigma after successful local search
                        sigma *= 0.7

            if calls >= budget:
                break

            # Update mean
            x_new = np.zeros(n)
            for i in range(mu):
                x_new += weights[i] * pops[i]

            # Update covariance and step size
            if sigma > 1e-10:
                y_mean = (x_new - x) / sigma
            else:
                y_mean = np.zeros(n)

            pc = (1 - c_c) * pc + c_c * y_mean
            ps = np.sqrt(1 - (1 - c_c)**2 * (1 - c_sigma)) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean

            norm_ps = np.linalg.norm(ps)
            hsig = (norm_ps / chi_n) < (1.4 + 2 / (n + 1))

            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - hsig) * c_c * (2 - c_c) * C)
            for i in range(mu):
                d = pops[i] - x
                C += c_mu * weights[i] * np.outer(d, d)

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
            sigma = max(1e-6, min(sigma, 10.0))

            x = x_new

        # Final polishing: use remaining budget for tight local search
        remaining = budget - calls
        if remaining >= n + 1 and time.time() < deadline - 3:
            final_budget = min(remaining, 25)
            final_x, final_f, _ = nelder_mead(best_x.copy(), final_budget, step_scale=0.5)
            if final_f < best_fx:
                best_fx = final_f
                best_x = final_x

        return best_x, best_fx
# EVOLVE-BLOCK-END
