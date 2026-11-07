"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with Cholesky-based sampling and final Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = max(4, 4 + int(3 * np.log(dim)))
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

        # State
        mean = np.zeros(dim)
        sigma = 2.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)

        calls = 0
        best_x = mean.copy()
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

        # Precompute Cholesky factor of C (initially I)
        L = np.eye(dim)

        while calls < budget and time.time() < deadline:
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
            sorted_fvals = fvals[order]

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
                # If C is not positive definite, add small regularization
                C += 1e-10 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)

            # Convergence check
            if sigma < 1e-10:
                break

        # Phase 2: Local refinement with Nelder-Mead if budget remains
        remaining = budget - calls
        if remaining > 10 and time.time() < deadline - 5:
            nm_budget = min(remaining, 50)
            nm_calls = [0]

            def nm_func(x):
                nonlocal calls, best_x, best_f
                if nm_calls[0] >= nm_budget or time.time() > deadline:
                    return best_f
                nm_calls[0] += 1
                calls += 1
                xc = np.clip(x, -5, 5)
                fx = func(xc)
                if fx < best_f:
                    best_f = fx
                    best_x = xc.copy()
                return fx

            try:
                from scipy.optimize import minimize
                res = minimize(nm_func, best_x, method='Nelder-Mead',
                              options={'maxiter': nm_budget, 'xatol': 1e-8, 'fatol': 1e-10})
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
