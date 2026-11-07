"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES with adaptive restarts and multi-start for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        best_x = None
        best_f = float('inf')
        eval_count = 0
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time exceeded")
            
            x = np.asarray(x, dtype=float)
            fx = func(x)
            eval_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            return fx

        def cma_es_single(start_x, sigma, max_evals):
            """Run a single CMA-ES instance with proper updates."""
            nonlocal eval_count, best_x, best_f, deadline
            
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            x_mean = start_x.copy()
            C = np.eye(dim)
            
            # Update rates
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            
            # Memory
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
            
            # Initial population
            pop = []
            fvals = []
            
            for i in range(pop_size):
                z = np.random.randn(dim)
                x_i = x_mean + sigma * z
                x_i = np.clip(x_i, -5.0, 5.0)
                pop.append(x_i)
                fvals.append(wrapped_func(x_i))
                if eval_count >= max_evals or time.time() > deadline:
                    break
            
            if len(pop) < pop_size:
                return x_mean, min(fvals) if fvals else float('inf')
            
            # Sort by fitness
            order = np.argsort(fvals)
            pop = [pop[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Initial weighted mean
            x_mean = np.sum(np.array(pop[:mu]) * weights[:, None], axis=0)
            
            no_improve = 0
            best_local_f = fvals[0]
            
            while eval_count < max_evals and time.time() < deadline:
                # Update evolution paths
                d = np.sqrt(c_sigma * (5 + c_sigma))
                delta_x = (np.array(pop[:mu]) - x_mean) / (sigma + 1e-32)
                p_sigma = (1 - c_sigma) * p_sigma + d * np.sqrt(mu_eff) * (weights @ delta_x)
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                
                # Update p_c
                h_sigma = 1.0 if norm_p_sigma < (1.4 + c_sigma) * chi_n else 0.0
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - start_x) / (sigma + 1e-32)
                
                # Update covariance matrix
                delta_h = (1 - c_c) ** 2 - (1 - c_c)
                C = (1 + c_1 + c_mu - c_1 - c_c * (1 + c_1 - c_c) + delta_h) * C
                C += c_1 * (np.outer(p_c, p_c) - C)
                rank_mu_term = np.sum(weights[:, None] * np.outer(delta_x, delta_x), axis=0)
                C += c_mu * rank_mu_term
                
                # Ensure positive definiteness
                C = 0.5 * (C + C.T)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = 0.5 * (C + C.T)
                
                # Compute square root via eigendecomposition
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)
                
                # Generate new population
                new_pop = []
                new_fvals = []
                
                for i in range(pop_size):
                    z = np.random.randn(dim)
                    x_i = x_mean + sigma * C_sqrt @ z
                    x_i = np.clip(x_i, -5.0, 5.0)
                    new_pop.append(x_i)
                    new_fvals.append(wrapped_func(x_i))
                    if eval_count >= max_evals or time.time() > deadline:
                        break
                
                if len(new_pop) < pop_size:
                    break
                
                # Sort
                order = np.argsort(new_fvals)
                new_pop = [new_pop[i] for i in order]
                new_fvals = [new_fvals[i] for i in order]
                
                # Update mean
                x_mean = np.sum(np.array(new_pop[:mu]) * weights[:, None], axis=0)
                
                # Check for improvement
                if new_fvals[0] < best_local_f - 1e-10:
                    no_improve = 0
                    best_local_f = new_fvals[0]
                else:
                    no_improve += 1
                
                if no_improve > 15:
                    break
                
                pop = new_pop
                fvals = new_fvals
            
            return x_mean, min(fvals) if fvals else float('inf')

        # Phase 1: Initial sampling with Sobol + strategic points
        try:
            from scipy.stats import qmc
            n_scout = 25
            sampler = qmc.Sobol(d=dim, scramble=True)
            samples = sampler.random(n=n_scout)
            samples = samples * 10.0 - 5.0
            scout_points = list(samples)
        except ImportError:
            n_scout = 25
            scout_points = np.random.uniform(-5, 5, size=(n_scout, dim)).tolist()

        # Add strategic points
        scout_points.append(np.zeros(dim))
        scout_points.append(np.ones(dim))
        scout_points.append(-np.ones(dim))
        scout_points.append(2.0 * np.ones(dim))
        scout_points.append(-2.0 * np.ones(dim))

        for p in scout_points:
            if time.time() > deadline or eval_count >= budget:
                break
            try:
                fp = wrapped_func(p)
            except Exception:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
                eval_count += 1
            except Exception:
                pass

        # Phase 2: Multi-start CMA-ES with adaptive restarts
        remaining = budget - eval_count
        if remaining > 50:
            # First CMA-ES run from best point
            try:
                x_cma, f_cma = cma_es_single(best_x, 1.5, max_evals=budget)
                if f_cma < best_f:
                    best_f = f_cma
                    best_x = x_cma.copy()
            except Exception:
                pass
        
        # Adaptive restarts: if we have budget left, try new starts
        remaining = budget - eval_count
        if remaining > 30:
            # Try a few more starts from different regions
            for restart in range(3):
                if time.time() > deadline or eval_count >= budget - 10:
                    break
                # Generate a new random start
                new_start = np.random.uniform(-4, 4, size=dim)
                try:
                    x_cma, f_cma = cma_es_single(new_start, 1.0, max_evals=budget)
                    if f_cma < best_f:
                        best_f = f_cma
                        best_x = x_cma.copy()
                except Exception:
                    break
        
        # Phase 3: Local polish with L-BFGS-B if budget remains
        remaining = budget - eval_count
        if remaining > 15 and best_x is not None:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 30, 'maxfun': remaining, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Phase 4: Final Nelder-Mead polish
        remaining = budget - eval_count
        if remaining > 8 and best_x is not None:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'maxfev': remaining, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
