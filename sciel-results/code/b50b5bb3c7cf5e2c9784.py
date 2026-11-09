"""Optimizer using CMA-ES with adaptive local polishing and restarts for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with adaptive local search and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin from 330s limit
        n = self.dim
        budget = self.budget

        # Global state
        best_x = None
        best_fx = np.inf
        total_calls = 0

        # CMA-ES Parameters tuned for low dimensions (n=5)
        # We use lambda=10, mu=5 for a balance of exploration and exploitation
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

        rng = np.random.default_rng(42)

        def nelder_mead(x0, max_evals, current_sigma):
            """Nelder-Mead local polishing."""
            simplex = [x0.copy() for _ in range(n + 1)]
            # Scale step by current sigma but ensure minimum step
            step = max(0.1, 0.5 * current_sigma)
            for i in range(n):
                simplex[i+1] = x0.copy()
                simplex[i+1][i] += step
            
            fvals = np.array([func(x) for x in simplex])
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

        # Main Loop with Restart Capability
        restart_count = 0
        max_restarts = 5 # Limit restarts to avoid infinite loops
        local_search_budget = 30
        stagnation_threshold = 8 # Generations without improvement before trying LS

        while total_calls < budget and restart_count <= max_restarts:
            if time.time() > deadline - 10:
                break

            # Initialize CMA-ES state
            # Start from the current best if available, otherwise origin
            if best_x is not None:
                x = best_x.copy()
                # Reduce sigma for restarts to be more local, but not too small
                sigma = max(0.5, 1.0 / (1 + restart_count)) 
            else:
                x = np.zeros(n)
                sigma = 1.0

            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            
            no_improve = 0
            generation = 0
            restart_needed = False

            # Run CMA-ES until budget is exhausted or stagnation triggers restart
            while total_calls < budget:
                if time.time() > deadline - 5:
                    break

                # Generate population
                pops = []
                fits = []

                for i in range(lambda_pop):
                    if total_calls >= budget or time.time() > deadline - 2:
                        break
                    z = rng.standard_normal(n)
                    try:
                        L = np.linalg.cholesky(C)
                        y = np.dot(L, z)
                    except:
                        y = z
                    x_i = np.clip(x + sigma * y, -5, 5)
                    f_i = func(x_i)
                    total_calls += 1
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

                # Check for Stagnation and Degeneration
                # 1. Stagnation: No improvement for many generations
                # 2. Degeneration: Sigma too small or Condition number too high
                cond_C = np.linalg.cond(C)
                
                # Trigger Local Search if stagnant
                if no_improve >= stagnation_threshold and total_calls + local_search_budget < budget:
                    if time.time() < deadline - 10:
                        local_x, local_f, used_evals = nelder_mead(best_x.copy(), local_search_budget, sigma)
                        total_calls += used_evals

                        if local_f < best_fx:
                            best_fx = local_f
                            best_x = local_x
                            no_improve = 0
                            # If LS found a better point, update CMA-ES mean to continue from there
                            x = best_x.copy()
                            # Reset stagnation counter
                        else:
                            # LS failed to improve. This is a strong signal to restart.
                            # Especially if we've been stagnant for a while.
                            if no_improve > stagnation_threshold * 2:
                                restart_needed = True
                                break

                # Check for degeneration (Sigma collapse or extreme anisotropy without progress)
                if sigma < 1e-4 or (cond_C > 1e6 and no_improve > 5):
                    if no_improve > 10:
                        restart_needed = True
                        break

                if total_calls >= budget:
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

            # If we broke out of the inner loop due to restart_needed and we have budget left
            if restart_needed and total_calls < budget:
                restart_count += 1
                # Optional: Small random perturbation of best_x to help restart
                # But CMA-ES with fresh C=I and reduced sigma should be sufficient
                continue
            else:
                # Budget exhausted or no restart needed
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
