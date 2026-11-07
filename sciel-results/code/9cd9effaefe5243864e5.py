# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES optimizer for BBOB f7 (Step Ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))  # For dim=5: 8
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

        rng = np.random.default_rng(42)
        
        calls = 0
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls += 1

        # Multi-start strategy:
        # We will run several CMA-ES instances with different initial means.
        # Each instance gets a portion of the budget.
        # If one finds the optimum (flat region), it will converge quickly.
        
        # Determine number of restarts.
        # With budget 500 and dim 5, lam=8, one full run is ~60 generations.
        # Let's try 5 restarts, each with budget/5 = 100 calls.
        # Or dynamically: if stagnation, restart.
        
        # Let's stick to a fixed number of restarts to ensure coverage.
        num_restarts = 5
        budget_per_restart = max(50, budget // num_restarts)
        
        for r in range(num_restarts):
            if calls >= budget or time.time() > deadline:
                break
            
            # Initialize CMA-ES state for this restart
            # Random start in [-4, 4]^dim
            mean = rng.uniform(-4, 4, size=dim)
            sigma = 1.0  # Start with smaller sigma to be more local? Or 2.0?
            # For step ellipsoidal, the "step" is at 0 in y-space.
            # The flat region is an orthant.
            # A larger sigma helps jump into the flat region if we are outside.
            # But if we are inside, we want to stay there.
            # Since we don't know where we are, let's use a moderate sigma.
            sigma = 1.5
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)

            # Run CMA-ES for this restart
            while calls < budget and time.time() < deadline:
                if sigma < 1e-12:
                    break

                # Generate population
                val, vec = np.linalg.eigh(C)
                val = np.clip(val, 1e-12, None)
                sqrt_val = np.sqrt(val)
                inv_sqrt_C = vec @ np.diag(1.0 / np.sqrt(val)) @ vec.T

                pop = np.empty((lam, dim))
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    pop[i] = mean + sigma * (vec @ (z * sqrt_val))
                    np.clip(pop[i], -5, 5, out=pop[i])

                # Evaluate population
                fvals = np.empty(lam)
                for i in range(lam):
                    if calls >= budget or time.time() > deadline:
                        fvals[i] = np.inf
                        continue
                    fvals[i] = func(pop[i])
                    calls += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = pop[i].copy()

                if calls >= budget or time.time() > deadline:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                sorted_pop = pop[order]
                sorted_fvals = fvals[order]

                # Update mean
                new_mean = np.sum(weights[:, np.newaxis] * sorted_pop[:mu], axis=0)

                # Update evolution paths
                diff = (new_mean - mean) / sigma
                pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * (inv_sqrt_C @ diff)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ diff)

                # Update sigma
                norm_ps = np.linalg.norm(ps)
                delta_sigma = (norm_ps / chi_n - 1) * d_sigma
                sigma *= np.exp(c_sigma * delta_sigma)
                sigma = np.clip(sigma, 1e-12, 10.0)

                # Update covariance matrix
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

                # Early stopping for this restart if converged
                if sigma < 1e-6:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
