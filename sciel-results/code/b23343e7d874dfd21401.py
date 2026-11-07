"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Robust CMA-ES with restarts for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(dim))  # 8 for dim=5
        mu = lam // 2                    # 4
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (2 * (mu_eff - 2 + 2 / mu_eff)) / ((dim + 2) ** 2 + mu_eff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        calls = 0
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls += 1

        def _eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Restart-based CMA-ES
        # We run multiple CMA-ES runs, restarting if stagnation is detected.
        # Initial sigma is large to cover the space.
        
        for restart in range(10): # Max 10 restarts
            if calls >= budget or time.time() > deadline - 5.0:
                break
            
            # Initialize or re-initialize CMA-ES state
            # On first run, start at 0. On restarts, start at best_x found so far.
            if restart == 0:
                mean = np.zeros(dim)
                sigma = 2.0
            else:
                mean = best_x.copy()
                # Shrink sigma on restart to refine around the best point,
                # but keep it large enough to escape local minima if the step is nearby.
                sigma = 0.5 * (2.0 / (restart + 1)) if restart > 0 else 2.0
                # Actually, for step functions, a larger sigma might be better to jump over steps.
                # Let's try a moderate sigma for restarts.
                sigma = 1.0 / (restart + 1) + 0.1

            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            
            # Run CMA-ES for a fixed number of generations or until budget/time
            # Each generation uses lam calls.
            max_generations = (budget - calls) // lam
            if max_generations <= 0:
                break
                
            # Limit generations per restart to allow for multiple restarts
            # If we have 500 calls, lam=8, that's ~60 generations total.
            # Let's allow up to 20 generations per restart.
            gens_this_restart = min(max_generations, 20)
            
            for gen in range(gens_this_restart):
                if calls >= budget or time.time() > deadline - 5.0:
                    break
                
                # Generate population
                pop = []
                # Pre-compute eigenvectors for efficiency? No, C changes every gen.
                val, vec = np.linalg.eigh(C)
                val = np.clip(val, 1e-10, None)
                sqrt_val = np.sqrt(val)
                
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    x = mean + sigma * (vec @ (z * sqrt_val))
                    pop.append(x)
                
                # Evaluate population
                fvals = []
                for x in pop:
                    fx = _eval(x)
                    if fx is None:
                        fvals.append(np.inf)
                    else:
                        fvals.append(fx)
                
                if calls >= budget or time.time() > deadline - 5.0:
                    break
                    
                fvals = np.array(fvals)
                order = np.argsort(fvals)
                sorted_pop = np.array([pop[i] for i in order])
                
                # Update mean
                new_mean = np.sum(weights[:, np.newaxis] * sorted_pop[:mu], axis=0)
                
                # Compute C^{-1/2}
                val, vec = np.linalg.eigh(C)
                val = np.clip(val, 1e-10, None)
                inv_sqrt_C = vec @ np.diag(1.0 / np.sqrt(val)) @ vec.T
                
                diff = (new_mean - mean) / sigma
                
                # Update paths
                pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * (inv_sqrt_C @ diff)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ diff)
                
                # Update sigma
                norm_ps = np.linalg.norm(ps)
                delta_sigma = (norm_ps / chi_n - 1) * d_sigma
                sigma *= np.exp(c_sigma * delta_sigma)
                sigma = np.clip(sigma, 1e-8, 5.0)
                
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
                
                # Stagnation check: if sigma is very small, break to restart
                if sigma < 1e-5:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
