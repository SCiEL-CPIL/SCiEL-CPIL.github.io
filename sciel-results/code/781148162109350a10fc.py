"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with adaptive restart and final coordinate-descent polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls += 1

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        def _eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def run_cmaes(mean_init, sigma_init, max_calls):
            """Run CMA-ES for up to max_calls additional evaluations."""
            nonlocal calls, best_x, best_f, deadline

            # CMA-ES parameters for dim=5
            lam = 10  # slightly larger population for better exploration
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = (2 * (mu_eff - 2 + 2 / mu_eff)) / ((dim + 2) ** 2 + mu_eff)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            mean = mean_init.copy()
            sigma = sigma_init
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            L = np.eye(dim)

            prev_best = best_f
            stagnation_counter = 0

            while calls < budget and calls < max_calls and time.time() < deadline:
                # Generate population using Cholesky factor
                pop = []
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    x = mean + sigma * (L @ z)
                    x = np.clip(x, -5, 5)
                    pop.append(x)

                # Evaluate population
                fvals = []
                for x in pop:
                    fx = _eval(x)
                    if fx is None:
                        fvals.append(np.inf)
                    else:
                        fvals.append(fx)

                if calls >= budget or time.time() > deadline:
                    break

                # Sort by fitness
                fvals = np.array(fvals)
                order = np.argsort(fvals)
                sorted_pop = np.array([pop[i] for i in order])

                # Update mean
                new_mean = np.sum(weights[:, np.newaxis] * sorted_pop[:mu], axis=0)

                # Compute C^{-1/2} via eigendecomposition for path updates
                val, vec = np.linalg.eigh(C)
                val = np.clip(val, 1e-10, None)
                inv_sqrt_C = vec @ np.diag(1.0 / np.sqrt(val)) @ vec.T

                diff = (new_mean - mean) / sigma

                # Update evolution paths
                pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * (inv_sqrt_C @ diff)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ diff)

                # Update sigma
                norm_ps = np.linalg.norm(ps)
                delta_sigma = (norm_ps / chi_n - 1) * d_sigma
                sigma *= np.exp(c_sigma * delta_sigma)
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update C
                y = (sorted_pop[:mu] - mean) / sigma
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(y[i], y[i])

                decay = 1 - c_1 - c_mu
                rank_one = np.outer(pc, pc) + (delta_sigma * c_sigma * (2 - c_sigma)) * C
                C = decay * C + c_1 * rank_one + c_mu * rank_mu
                C = (C + C.T) / 2

                # Update mean
                mean = new_mean

                # Update Cholesky factor for next iteration
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-10 * np.eye(dim)
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(dim)

                # Convergence check
                if sigma < 1e-10:
                    break

                # Stagnation detection: if no improvement in 30 calls, restart
                if best_f < prev_best - 1e-10:
                    prev_best = best_f
                    stagnation_counter = 0
                else:
                    stagnation_counter += lam

                if stagnation_counter > 100 and calls < budget - 50:
                    # Restart with larger sigma to escape local minimum
                    sigma = max(sigma * 2.0, 1.0)
                    mean = best_x.copy()
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                    C = np.eye(dim)
                    L = np.eye(dim)
                    stagnation_counter = 0
                    prev_best = best_f
                    break

            return best_x, best_f

        # Phase 1: Initial CMA-ES run
        max_calls_phase1 = budget - 50
        run_cmaes(np.zeros(dim), 2.0, max_calls_phase1)

        # Phase 2: Coordinate descent polish
        if calls < budget and time.time() < deadline - 5:
            # Coordinate descent with adaptive step size
            step_sizes = [1.0, 0.1, 0.01, 0.001, 0.0001]
            x = best_x.copy()
            f = best_f

            for ss in step_sizes:
                for _ in range(10):  # 10 passes per step size
                    if calls >= budget or time.time() > deadline:
                        break
                    for d in range(dim):
                        if calls >= budget or time.time() > deadline:
                            break
                        # Try positive and negative steps
                        for sign in [1, -1]:
                            if calls >= budget or time.time() > deadline:
                                break
                            x_new = x.copy()
                            x_new[d] += sign * ss
                            x_new = np.clip(x_new, -5, 5)
                            fx = _eval(x_new)
                            if fx is None:
                                break
                            if fx < f - 1e-12:
                                x = x_new.copy()
                                f = fx
                                best_x = x.copy()
                                best_f = f

        return best_x, best_f

# EVOLVE-BLOCK-END
