"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Global state
        global_best_x = np.zeros(dim)
        global_best_f = float('inf')
        calls = 0
        start_time = time.time()
        max_time = 280.0 # Safety margin for 330s limit

        def can_continue():
            return calls < budget and (time.time() - start_time) < max_time

        # CMA-ES Parameters
        # For dim=5, lambda=10 is a good tradeoff between exploration and exploitation
        lambda_pop = max(4 + int(3 * np.log(dim)), 10) 
        mu = lambda_pop // 2
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

        def run_cma_es(x0, sigma0, max_iter):
            nonlocal global_best_x, global_best_f, calls
            
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            Csqrt_inv = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Precompute Cholesky factor of C (identity initially)
            B = np.eye(dim)
            D = np.ones(dim)
            
            for gen in range(max_iter):
                if not can_continue():
                    break
                
                # Generate population
                y = np.random.default_rng(42 + gen).standard_normal((lambda_pop, dim))
                x_pop = x_mean + sigma * (y @ B.T * D)
                
                # Evaluate population
                f_pop = []
                for i in range(lambda_pop):
                    if not can_continue():
                        break
                    x_i = np.clip(x_pop[i], -5, 5)
                    f_i = func(x_i)
                    calls += 1
                    f_pop.append(f_i)
                    
                    if f_i < global_best_f:
                        global_best_f = f_i
                        global_best_x = x_i.copy()
                
                if len(f_pop) < lambda_pop:
                    break
                    
                f_pop = np.array(f_pop)
                order = np.argsort(f_pop)
                x_sorted = x_pop[order]
                f_sorted = f_pop[order]
                
                # Update mean
                x_new = np.sum(weights[:, np.newaxis] * x_sorted[:mu], axis=0)
                
                # Update covariance matrix
                diff = x_sorted[:mu] - x_mean
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - cc) * C) + cmu * (diff.T @ (weights[:, np.newaxis] * diff))
                
                # Symmetrize C to prevent numerical drift
                C = (C + C.T) / 2
                
                # Update evolution paths
                pc = (1 - cc) * pc + cc * np.sqrt(mueff) / (sigma * np.sqrt(np.diag(C))) @ (x_new - x_mean)
                
                # Approximate inverse square root of C for step size adaptation
                # We use a simplified update or just track the norm of the step
                # Standard CMA-ES uses C^-1/2. For dim=5, we can do Cholesky periodically or use eigendecomp if slow.
                # Given dim=5, eigendecomp is fast enough for 50 generations.
                eigval, eigvec = np.linalg.eigh(C)
                # Ensure positive definiteness
                eigval = np.maximum(eigval, 1e-12)
                D = np.sqrt(eigval)
                B = eigvec
                Csqrt = B * np.sqrt(D)
                Csqrt_inv = B / np.sqrt(D)
                
                # Update step size (CSA)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * Csqrt_inv @ (x_new - x_mean) / sigma
                chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim * dim))
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update mean
                x_mean = x_new
                
                # Check termination conditions
                if sigma < 1e-12:
                    break
                # If we have made very little progress, break early to allow restart
                # This is handled by the outer loop restart logic
                
            return x_mean, sigma

        # Main Loop: Multi-start CMA-ES
        # Start with a large sigma to cover the domain
        # Domain is [-5, 5]^5, diagonal is ~14.14. Start sigma ~ 2.5 or 3.0
        current_sigma = 3.0
        restart_count = 0
        max_restarts = 10
        min_sigma_restart = 0.05
        
        while can_continue() and restart_count < max_restarts:
            # Start point: best found so far, or random if first
            if restart_count == 0:
                x0 = np.random.default_rng(42).uniform(-2, 2, dim)
            else:
                # Jitter the best point to escape local minima
                x0 = global_best_x + np.random.default_rng(42 + restart_count).normal(0, 0.5, dim)
                x0 = np.clip(x0, -5, 5)
            
            # Run CMA-ES
            # Budget allocation: give each restart a chunk of the remaining budget
            # Or just run until budget is exhausted in the last restart
            # Simple strategy: run CMA-ES with a fixed number of generations based on remaining budget
            remaining_budget = budget - calls
            if remaining_budget <= 0:
                break
            
            # Generations = remaining / lambda
            max_gen = remaining_budget // lambda_pop
            if max_gen < 5:
                max_gen = 5 # Minimum generations to make progress
            
            x_mean, final_sigma = run_cma_es(x0, current_sigma, max_gen)
            
            # If sigma became very small, we likely converged to a local minimum.
            # Restart with a larger sigma from the best point.
            if final_sigma < min_sigma_restart:
                restart_count += 1
                # Increase sigma for next restart to explore wider
                current_sigma = max(0.5, current_sigma * 0.8) 
            else:
                # If we ran out of budget, we are done
                break
            # If we didn't converge but used up budget, break
            if not can_continue():
                break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
