"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES with adaptive population sizing for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _run_cmaes(self, func, x0, sigma0, budget, dim, rng, deadline):
        """Run CMA-ES from a given starting point."""
        # Adaptive population size: smaller for larger budgets to save evaluations
        lam = max(4, int(4 + 3 * np.log(dim)))
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

        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)

        calls = 0
        best_x = mean.copy()
        best_f = func(best_x)
        calls += 1

        if calls >= budget or time.time() > deadline:
            return best_x, best_f, calls

        def _eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return None
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        L = np.eye(dim)
        stagnation_counter = 0
        best_f_prev = best_f
        stagnation_limit = 30  # restart after 30 generations without improvement

        while calls < budget and time.time() < deadline:
            # Generate population
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

            fvals = np.array(fvals)
            order = np.argsort(fvals)
            sorted_pop = np.array([pop[i] for i in order])
            sorted_fvals = fvals[order]

            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * sorted_pop[:mu], axis=0)

            # Compute C^{-1/2} via eigendecomposition
            val, vec = np.linalg.eigh(C)
            val = np.clip(val, 1e-12, None)
            inv_sqrt_C = vec @ np.diag(1.0 / np.sqrt(val)) @ vec.T

            diff = (new_mean - mean) / sigma

            # Update evolution paths
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * (inv_sqrt_C @ diff)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ diff)

            # Update sigma
            norm_ps = np.linalg.norm(ps)
            delta_sigma = (norm_ps / chi_n - 1) * d_sigma
            sigma *= np.exp(c_sigma * delta_sigma)
            sigma = np.clip(sigma, 1e-12, 10.0)

            # Update C
            y = (sorted_pop[:mu] - mean) / sigma
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(y[i], y[i])

            decay = 1 - c_1 - c_mu
            rank_one = np.outer(pc, pc) + (delta_sigma * c_sigma * (2 - c_sigma)) * C
            C = decay * C + c_1 * rank_one + c_mu * rank_mu
            C = (C + C.T) / 2

            mean = new_mean

            # Update Cholesky factor
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-10 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)

            # Convergence check
            if sigma < 1e-12:
                break

            # Stagnation detection
            if best_f < best_f_prev - 1e-10:
                stagnation_counter = 0
                best_f_prev = best_f
            else:
                stagnation_counter += 1

            if stagnation_counter >= stagnation_limit:
                break

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin

        rng = np.random.default_rng(123)
        dim = self.dim
        budget = self.budget

        # Phase 1: Multi-start CMA-ES with diverse initial points
        # Use a portion of budget for multiple restarts
        n_restarts = 5
        budget_per_restart = max(50, budget // n_restarts)

        # Initial points: center and diverse random points
        x0_list = [np.zeros(dim)]
        for i in range(n_restarts - 1):
            x0 = (rng.uniform(-4, 4, size=dim))
            x0_list.append(x0)

        global_best_x = None
        global_best_f = np.inf
        total_calls = 0

        for i, x0 in enumerate(x0_list):
            if total_calls >= budget or time.time() > deadline:
                break

            remaining = budget - total_calls
            # Use decreasing budget for later restarts
            alloc = min(budget_per_restart, remaining)
            if alloc < 20:
                break

            sigma0 = 2.0 if i == 0 else 1.0

            best_x, best_f, calls = self._run_cmaes(
                func, x0, sigma0, alloc, dim, rng, deadline
            )

            total_calls += calls

            if best_f < global_best_f:
                global_best_f = best_f
                global_best_x = best_x.copy()

        # Phase 2: Final local refinement using a small CMA-ES around the best point
        remaining = budget - total_calls
        if remaining > 20 and time.time() < deadline - 5 and global_best_x is not None:
            # Use a very small sigma for local refinement
            sigma_final = 0.1
            best_x, best_f, calls = self._run_cmaes(
                func, global_best_x, sigma_final, min(remaining, 50), dim, rng, deadline
            )
            if best_f < global_best_f:
                global_best_f = best_f
                global_best_x = best_x.copy()

        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
