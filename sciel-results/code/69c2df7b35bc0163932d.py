"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid optimizer for Lunacek bi-Rastrigin (f24).
    Phase 1: Global search using Latin Hypercube Sampling.
    Phase 2: Local exploitation using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin for 300s limit

        global_best = None
        global_fx = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, global_best, global_fx
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < global_fx:
                global_fx = fx
                global_best = x.copy()
            return fx

        # --- Phase 1: Global Search ---
        # Allocate 20% of budget to global search to find the correct basin
        n_global = 100
        rng = np.random.default_rng(42)
        
        # LHS points in [-4.5, 4.5]^dim
        # The optimum is guaranteed to be in [-4, 4]^5, so sampling slightly wider is safe
        lhs_points = np.zeros((n_global, dim))
        for d in range(dim):
            edges = np.linspace(-4.5, 4.5, n_global + 1)
            for i in range(n_global):
                lhs_points[i, d] = rng.uniform(edges[i], edges[i+1])
        
        # Shuffle to ensure diversity in evaluation order
        perm = rng.permutation(n_global)
        lhs_points = lhs_points[perm]

        for i in range(n_global):
            if calls_made >= budget - 50 or time.time() > deadline - 10:
                break
            evaluate(lhs_points[i])

        # --- Phase 2: Local Search with CMA-ES ---
        if global_best is not None and calls_made < budget - 20 and time.time() < deadline - 10:
            x0 = global_best.copy()
            fx0 = global_fx
            
            # CMA-ES Parameters
            # Population size
            lambda_pop = 4 + int(3 * np.log(dim))
            if lambda_pop < 2: lambda_pop = 2
            if lambda_pop > 20: lambda_pop = 20 # Cap for small dim
            
            # Weights
            weights = np.log(lambda_pop / 2 + 0.5) - np.log(np.arange(1, lambda_pop + 1))
            weights /= np.sum(weights)
            mu = int(lambda_pop / 2)
            mueff = 1.0 / np.sum(weights[:mu] ** 2)
            
            # Adaptation rates
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
            
            # Chi lambda
            chi_lambda = np.sqrt(lambda_pop) * (1 - 1 / (4 * lambda_pop) + 1 / (25 * lambda_pop ** 2))
            
            # Initial state
            x_mean = x0.copy()
            sigma = 1.0 # Initial step size
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            # Precompute inverse square root of C
            def update_C_inv_sqrt(C):
                nonlocal C_inv_sqrt
                # Use eigendecomposition for stability
                D, V = np.linalg.eigh(C)
                # Clip small eigenvalues
                D = np.clip(D, 1e-12, None)
                C_inv_sqrt = (V * (1 / np.sqrt(D))) @ V.T
                return C_inv_sqrt
            
            update_C_inv_sqrt(C)
            
            # CMA-ES Main Loop
            while calls_made < budget - 5 and time.time() < deadline - 5:
                # Generate population
                Y = rng.standard_normal((lambda_pop, dim))
                X = np.zeros((lambda_pop, dim))
                F = np.zeros(lambda_pop)
                
                for i in range(lambda_pop):
                    if calls_made >= budget - 5 or time.time() > deadline - 5:
                        break
                    X[i] = x_mean + sigma * (C_inv_sqrt @ Y[i])
                    F[i] = evaluate(X[i])
                
                if i < lambda_pop - 1:
                    # If we broke early, adjust lambda for this iteration
                    lambda_pop_current = i + 1
                    if lambda_pop_current < 2:
                        break
                    # Re-normalize weights for the smaller population? 
                    # For simplicity and robustness, if we break early, we just stop CMA-ES 
                    # or use the partial population. Given the tight budget, we might just break.
                    # However, to maximize usage, let's just break the loop if we didn't finish the population.
                    break

                # Sort by fitness
                order = np.argsort(F)
                X = X[order]
                F = F[order]
                
                # Update mean
                x_mean_new = np.sum(weights[:mu] * X[:mu], axis=0)
                
                # Update sigma (CSA)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C_inv_sqrt @ (x_mean_new - x_mean))
                
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made // lambda_pop))) < (chi_lambda + 1.4) * np.sqrt((dim + 1) / (dim - 1)):
                    sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_lambda - 1))
                    
                # Update covariance (CSA)
                pc = (1 - c1) * pc + c1 * (x_mean_new - x_mean) / sigma
                
                rank_mu = np.outer(weights[:mu], np.ones((mu, 1))) @ X[:mu]
                rank_mu = np.outer(rank_mu - x_mean, rank_mu - x_mean) / sigma**2
                rank_one = np.outer(pc, pc)
                
                C = (1 - c1 - cmu) * C + c1 * (rank_one + (c1 / (1 - c1)) * C) + cmu * rank_mu
                
                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                
                # Update inverse square root
                update_C_inv_sqrt(C)
                
                x_mean = x_mean_new
                
                # Convergence check
                if np.max(F) - np.min(F) < 1e-8 * (1 + abs(np.min(F))):
                    break
                if sigma < 1e-10:
                    break

        return global_best, global_fx
# EVOLVE-BLOCK-END
