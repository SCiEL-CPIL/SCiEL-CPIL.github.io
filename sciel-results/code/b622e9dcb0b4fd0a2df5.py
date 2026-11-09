"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 10, time guard, and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin
        budget = self.budget
        dim = self.dim

        # Global best tracking
        global_best_x = None
        global_best_f = np.inf

        def update_global(x, f):
            nonlocal global_best_x, global_best_f
            if f < global_best_f:
                global_best_x = x.copy()
                global_best_f = f

        # CMA-ES implementation
        def cmaes(func, x0, sigma0, budget, deadline):
            n = dim
            # Population size
            lam = 10
            mu = lam // 2  # number of selected parents

            # Weights
            w = np.log(np.arange(mu, 0, -1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            # Strategy parameters
            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = (2 / ((n + 1.3) ** 2 + mu_eff)) * (2 - cc)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            # Initial state
            x_mean = x0.astype(float).copy()
            sigma = sigma0
            C = np.eye(n)
            C_sqrt = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Best tracking
            best_x = x0.copy()
            best_f = func(best_x)
            calls = 1

            # History for restart detection
            no_improve = 0

            while calls < budget and time.time() < deadline:
                # Generate population
                z = np.random.default_rng(np.random.randint(0, 2**31 - 1)).standard_normal((lam, n))
                # Use CMA-ES sampling: x_i = x_mean + sigma * C_sqrt @ z_i
                pop = x_mean + sigma * (z @ C_sqrt.T)
                # Clip to box
                pop = np.clip(pop, -5, 5)

                # Evaluate population
                fvals = np.empty(lam)
                for i in range(lam):
                    fvals[i] = func(pop[i])
                    calls += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = pop[i].copy()
                        no_improve = 0
                    else:
                        no_improve += 1

                # Update global best
                if best_f < global_best_f:
                    global_best_x = best_x.copy()
                    global_best_f = best_f

                # Sort by fitness
                order = np.argsort(fvals)
                pop_sorted = pop[order]
                fvals_sorted = fvals[order]

                # Check for stagnation: if no improvement in last 50*lam evaluations, restart
                if no_improve > 50 * lam:
                    break

                # Update mean
                x_new = np.sum(w[:, None] * pop_sorted[:mu], axis=0)

                # Update path
                inv_C_sqrt = np.linalg.inv(C_sqrt)
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * inv_C_sqrt @ (x_new - x_mean) / sigma
                p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * (1 - (1 - cc) ** 2)) * (x_new - x_mean) / sigma

                hsig = np.linalg.norm(p_sigma) / chi_n < (1.4 + 2 / (n + 1)) * (1 - cs)

                # Update covariance
                C = (1 + c1 * (2 - c1) + cmu * (hsig - 1) - c1) * C
                C += c1 * (np.outer(p_c, p_c) - C)
                C += cmu * np.sum(w[:, None, None] * np.outer(pop_sorted[:mu] - x_mean, pop_sorted[:mu] - x_mean), axis=0) / sigma ** 2

                # Ensure symmetry and positive semi-definite
                C = (C + C.T) / 2
                # Renormalize to keep trace constant
                trace = np.trace(C)
                if trace > 0:
                    C = C * (n / trace)

                # Update C_sqrt via eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Clamp sigma
                sigma = np.clip(sigma, 1e-10, 1e5)

                # Update mean
                x_mean = x_new

                # Safety: if sigma becomes too small, break
                if sigma < 1e-8:
                    break

            return best_x, best_f, calls

        # Multiple restarts with different initializations
        rng = np.random.default_rng(42)

        # First, do a quick random search to find a good starting point
        n_random = min(20, budget // 10)
        best_rand_x = None
        best_rand_f = np.inf
        for _ in range(n_random):
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            if f < best_rand_f:
                best_rand_f = f
                best_rand_x = x.copy()
        update_global(best_rand_x, best_rand_f)

        # CMA-ES restarts
        n_restarts = 3
        budget_per_restart = (budget - n_random) // n_restarts

        for r in range(n_restarts):
            if time.time() >= deadline:
                break

            # Initialize CMA-ES around best found so far, with some perturbation
            if r == 0:
                x0 = best_rand_x.copy()
                sigma0 = 1.0
            else:
                # Perturb the best found
                x0 = np.clip(best_rand_x + 0.5 * rng.standard_normal(dim), -5, 5)
                sigma0 = 0.5

            # Run CMA-ES
            bx, bf, calls_used = cmaes(func, x0, sigma0, budget_per_restart, deadline)
            update_global(bx, bf)

            # If we've used most of the budget, stop
            if budget_per_restart * (r + 1) + n_random >= budget - 10:
                break

        # Final polish: try a few small perturbations around the best
        if time.time() < deadline and global_best_x is not None:
            for _ in range(min(20, max(5, budget - n_random - n_restarts * budget_per_restart))):
                if time.time() >= deadline:
                    break
                x = np.clip(global_best_x + 0.1 * rng.standard_normal(dim), -5, 5)
                f = func(x)
                update_global(x, f)

        if global_best_x is None:
            global_best_x = np.zeros(dim)
            global_best_f = func(global_best_x)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
