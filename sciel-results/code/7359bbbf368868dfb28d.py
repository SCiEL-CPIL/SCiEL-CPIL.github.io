# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-start CMA-ES with dynamic restarts and final (1+1)-ES polish
    for BBOB f7 (Step Ellipsoidal).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))  # 8 for dim=5
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

        # Reserve 10% of budget for final polish if possible
        polish_budget = max(10, budget // 10)
        main_budget = budget - polish_budget
        
        # Multi-start loop
        # We will run restarts until we run out of main_budget or time.
        # Each restart gets a "slice" of budget, but we can terminate early if stagnant.
        
        max_restarts = 10
        min_calls_per_restart = 30 # Don't kill a restart too quickly
        
        for r in range(max_restarts):
            if calls >= main_budget or time.time() > deadline:
                break
            
            # Initialize CMA-ES state
            # Use diverse starting points
            if r == 0:
                mean = np.zeros(dim)
            else:
                mean = rng.uniform(-4, 4, size=dim)
            
            sigma = 2.0
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            
            calls_at_start = calls
            local_best_f = best_f
            
            # Run CMA-ES for this restart
            while calls < main_budget and time.time() < deadline:
                if sigma < 1e-10:
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
                    if calls >= main_budget or time.time() > deadline:
                        fvals[i] = np.inf
                        continue
                    fvals[i] = func(pop[i])
                    calls += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = pop[i].copy()

                if calls >= main_budget or time.time() > deadline:
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

                # Stagnation check: if no improvement in global best for 2*lam calls, restart
                if calls - calls_at_start > 2 * lam and best_f == local_best_f:
                    # If we haven't improved the global best in a while, break this restart
                    break

        # Final Polish Phase: (1+1)-ES around the global best
        # This helps refine the solution if CMA-ES stopped a bit early or sigma was too large.
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma_polish = 0.1 # Start small
            success_count = 0
            
            while calls < budget and time.time() < deadline:
                y = np.clip(x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma_polish *= 1.2 # Increase step size on success
                    success_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma_polish *= 0.8 # Decrease step size on failure
                    success_count = 0
                    
                # Stop polish if sigma becomes too small or we are stuck
                if sigma_polish < 1e-6:
                    break
                # Also stop if we've had many failures in a row (converged)
                if sigma_polish < 1e-3 and success_count == 0 and calls > main_budget + 10:
                    # Just let it run until budget is out, it's cheap
                    pass

        return best_x, best_f

# EVOLVE-BLOCK-END
