# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid Optimizer: Multi-start CMA-ES with Nelder-Mead polish for BBOB f7.
    
    Strategy:
    1. Use CMA-ES for global exploration and handling the rotated ellipsoid structure.
    2. Detect stagnation in CMA-ES (likely converged to a local min or stuck in flat region).
    3. If stagnation is detected or budget is nearly exhausted, switch to Nelder-Mead 
       for fine-tuning the best point found. Nelder-Mead is effective for the final 
       convergence in the smooth convex region of the Step Ellipsoidal function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls_made += 1

        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # CMA-ES Parameters
        # For dim=5, standard parameters are sufficient
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

        def run_cma_es(initial_mean, sigma_init, max_calls):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time.time() > deadline:
                return
            
            mean = initial_mean.copy()
            sigma = sigma_init
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            
            # Pre-compute inverse sqrt C for efficiency? 
            # We update C every generation, so we must recompute or use stable updates.
            # For dim=5, full decomposition is cheap.
            
            stagnant_count = 0
            last_best_f = best_f
            
            while calls_made < budget and time.time() < deadline:
                # Check if we should stop this CMA-ES run
                if sigma < 1e-12:
                    break
                
                # Generate population
                val, vec = np.linalg.eigh(C)
                val = np.clip(val, 1e-12, None)
                sqrt_val = np.sqrt(val)
                inv_sqrt_C = vec @ np.diag(1.0 / np.sqrt(val)) @ vec.T
                
                pop = np.empty((lam, dim))
                fvals = np.empty(lam)
                
                for i in range(lam):
                    if calls_made >= budget or time.time() > deadline:
                        fvals[i] = np.inf
                        continue
                    z = rng.standard_normal(dim)
                    x_i = mean + sigma * (vec @ (z * sqrt_val))
                    np.clip(x_i, -5, 5, out=x_i)
                    pop[i] = x_i
                    fvals[i] = func(x_i)
                    calls_made += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = x_i.copy()
                
                if calls_made >= budget or time.time() > deadline:
                    break

                # Sort
                order = np.argsort(fvals)
                sorted_pop = pop[order]
                sorted_fvals = fvals[order]
                
                # Update mean
                new_mean = np.sum(weights[:, np.newaxis] * sorted_pop[:mu], axis=0)
                
                # Update paths
                diff = (new_mean - mean) / sigma
                pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * (inv_sqrt_C @ diff)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ diff)
                
                # Update sigma
                norm_ps = np.linalg.norm(ps)
                delta_sigma = (norm_ps / chi_n - 1) * d_sigma
                sigma *= np.exp(c_sigma * delta_sigma)
                sigma = np.clip(sigma, 1e-12, 10.0)
                
                # Update covariance
                y = (sorted_pop[:mu] - mean) / sigma
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(y[i], y[i])
                
                decay = 1 - c_1 - c_mu
                rank_one = np.outer(pc, pc) + (delta_sigma * c_sigma * (2 - c_sigma)) * C
                C = decay * C + c_1 * rank_one + c_mu * rank_mu
                C = (C + C.T) / 2
                
                mean = new_mean
                
                # Stagnation check: If best_f hasn't improved in last 10 generations
                if best_f < last_best_f - 1e-10:
                    stagnant_count = 0
                    last_best_f = best_f
                else:
                    stagnant_count += 1
                
                # If stagnant for a while, break to allow polish or restart
                if stagnant_count > 15:
                    break

        # Multi-Start Strategy
        # We allocate 80% of budget to CMA-ES exploration and 20% to polish.
        # But we can be dynamic.
        
        # Start 1: Origin
        run_cma_es(np.zeros(dim), 1.5, int(budget * 0.4))
        
        # Start 2: Random
        if calls_made < budget * 0.8 and time.time() < deadline:
            mean2 = rng.uniform(-4, 4, size=dim)
            run_cma_es(mean2, 1.5, int(budget * 0.4))
            
        # Start 3: Another Random if budget remains
        if calls_made < budget * 0.8 and time.time() < deadline:
            mean3 = rng.uniform(-4, 4, size=dim)
            run_cma_es(mean3, 1.5, int(budget * 0.2))

        # Phase 2: Polish with Nelder-Mead
        # Use remaining budget
        if calls_made < budget and time.time() < deadline:
            # Nelder-Mead parameters
            # maxiter is limited by remaining budget
            remaining_calls = budget - calls_made
            # Nelder-Mead uses ~ (dim+1) calls per iteration roughly, 
            # but it's adaptive. Let's cap maxiter to be safe.
            max_iter_nm = max(10, remaining_calls // (dim + 1))
            
            # We need a wrapper to count calls
            nm_calls = [0]
            def nm_func(x):
                if nm_calls[0] >= remaining_calls or time.time() > deadline:
                    return 1e10
                nm_calls[0] += 1
                calls_made += 1
                return func(x)
            
            try:
                res = minimize(nm_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': max_iter_nm, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
