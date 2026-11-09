"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Correct CMA-ES optimizer for BBOB f17 (Schaffers F7)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        total_budget = self.budget
        calls_used = 0
        best_x = None
        best_f = np.inf

        def track_call(x, f):
            nonlocal calls_used, best_x, best_f
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def cma_es(rng, x0, sigma0):
            nonlocal calls_used, best_x, best_f, deadline
            n = dim
            # Population size: lambda = 4 + floor(3 ln n) for n=5 gives 4+4=8, use 10 for robustness
            m = 10
            mu = max(1, m // 2)
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w**2)
            
            # Strategy parameters
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
            
            # State variables
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            L = np.eye(n)  # Cholesky: C = L @ L.T
            ps = np.zeros(n)
            pc = np.zeros(n)
            chi_n = np.sqrt(n) * (1 + 1/(4*n) - 1/(21*n**2))
            
            # Initial population evaluation
            pop = np.empty((m, n))
            pop_f = np.empty(m)
            
            # Evaluate initial mean
            f_mean = func(mean)
            track_call(mean, f_mean)
            
            for i in range(m):
                if time.time() > deadline or calls_used >= total_budget:
                    return
                z = rng.standard_normal(n)
                xi = np.clip(mean + sigma * (L @ z), -5, 5)
                pop[i] = xi
                pop_f[i] = func(xi)
                track_call(xi, pop_f[i])
            
            # Main loop
            while calls_used + m <= total_budget and time.time() < deadline:
                order = np.argsort(pop_f)
                
                # New mean
                mean_new = (w[:mu, None] * pop[order[:mu], :]).sum(axis=0)
                
                # Step vectors in normalized space
                y_mean = (mean_new - mean) / sigma
                y_selected = (pop[order[:mu], :] - mean) / sigma
                
                # Update sigma path
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * np.linalg.solve(L.T, y_mean)
                
                # Delta_h for covariance update
                norm_ps = np.linalg.norm(ps)
                h_sigma = norm_ps / chi_n < (1.4 + 2/(n+1)) * (1 - (1-c_sigma)**10)
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                
                # Update correlation path
                pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_mean
                
                # Update covariance matrix
                C = (1 + c_1*delta_h - c_1 - c_mu) * C
                C += c_1 * (np.outer(pc, pc) - delta_h * C)
                
                # Rank-mu update: use whitened steps
                for i in range(mu):
                    # Whitened step: L.T^{-1} @ y_i
                    y_w = np.linalg.solve(L.T, y_selected[i])
                    C += c_mu * w[i] * np.outer(y_w, y_w)
                
                C = (C + C.T) / 2  # Enforce symmetry
                
                # Update sigma
                sigma *= np.exp((c_sigma / d_sigma) * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Update Cholesky
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-8 * np.eye(n)
                    L = np.linalg.cholesky(C)
                
                # Update mean
                mean = mean_new
                
                # Check if best improved
                if pop_f[order[0]] < best_f:
                    best_f = pop_f[order[0]]
                    best_x = pop[order[0]].copy()
                
                # Generate next population
                for i in range(m):
                    if time.time() > deadline or calls_used >= total_budget:
                        return
                    z = rng.standard_normal(n)
                    xi = np.clip(mean + sigma * (L @ z), -5, 5)
                    pop[i] = xi
                    pop_f[i] = func(xi)
                    track_call(xi, pop_f[i])

        rng = np.random.default_rng(42)
        
        # Phase 1: Brief random search to find a good starting point
        n_explore = min(15, total_budget // 8)
        for _ in range(n_explore):
            if time.time() > deadline or calls_used >= total_budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            track_call(x, fx)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            track_call(best_x, best_f)
        
        # Phase 2: Single focused CMA-ES run from best point
        if total_budget - calls_used > 15:
            cma_es(rng, best_x.copy(), 1.0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
