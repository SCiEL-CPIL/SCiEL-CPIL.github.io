"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid CMA-ES and L-BFGS-B optimizer for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s kill limit. Use 320s to be safe.
        deadline = start_time + 320.0

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # --- CMA-ES Parameters ---
        # Standard parameters for dim=5
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
            # Clip to bounds before calling to be safe, though func clips internally
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # --- Phase 1: CMA-ES ---
        # Run CMA-ES until we have used ~70% of the budget or time is low
        cma_budget_limit = int(budget * 0.7)
        
        while calls < cma_budget_limit and time.time() < deadline - 10.0:
            # Generate population
            pop = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                # Use eigendecomposition for sampling (dim=5 is small)
                val, vec = np.linalg.eigh(C)
                val = np.clip(val, 1e-10, None)
                x = mean + sigma * (vec @ (z * np.sqrt(val)))
                pop.append(x)
            
            # Evaluate population
            fvals = []
            for x in pop:
                fx = _eval(x)
                if fx is None:
                    fvals.append(np.inf)
                else:
                    fvals.append(fx)
            
            if calls >= cma_budget_limit or time.time() > deadline - 10.0:
                break
            
            # Sort
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
            
            mean = new_mean
            
            # Convergence check
            if sigma < 1e-8:
                break

        # --- Phase 2: L-BFGS-B Refinement ---
        remaining_calls = budget - calls
        if remaining_calls > 10 and time.time() < deadline - 5.0:
            # L-BFGS-B uses numerical gradients. 
            # Each function call in L-BFGS-B context might consume 1 budget call.
            # We pass the remaining budget as maxiter roughly, but we must track calls.
            
            lbfgs_calls = [0]
            lbfgs_budget = remaining_calls
            
            def lbfgs_func(x):
                nonlocal calls, best_x, best_f
                if lbfgs_calls[0] >= lbfgs_budget or calls >= budget or time.time() > deadline:
                    # Return a large value or the best known to stop
                    return best_f
                lbfgs_calls[0] += 1
                x_clipped = np.clip(x, -5.0, 5.0)
                fx = func(x_clipped)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x_clipped.copy()
                return fx

            try:
                # Use L-BFGS-B with bounds
                # maxiter is an upper bound on iterations, not calls.
                # We rely on the lbfgs_func guard to stop.
                res = minimize(lbfgs_func, best_x, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0)] * dim,
                               options={'maxiter': lbfgs_budget, 'gtol': 1e-8, 'ftol': 1e-12})
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
