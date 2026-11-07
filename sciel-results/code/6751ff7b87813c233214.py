"""Optimizer for BBOB-style f17 (Schaffers F7) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    
    Strategy:
    1. Initial sampling: Evaluate a few points (center + random) to find a good starting basin.
    2. CMA-ES: Run CMA-ES from the best initial point with the remaining budget.
       - Small population size to allow many generations (early progress).
       - Robust handling of the deadline.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # --- Phase 1: Initial Sampling ---
        # Evaluate center and a few random points in [-4, 4]^5
        # Minimizer is known to be in [-4, 4]^5
        n_random = 3
        points = [np.zeros(dim)]
        points.extend(rng.uniform(-4.0, 4.0, size=(n_random, dim)).tolist())
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        for p in points:
            if calls_made >= budget or time.time() > deadline:
                break
            f_val = func(p)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = p.copy()

        # If we ran out of budget, return best found
        if calls_made >= budget or best_x is None:
            return best_x, best_f

        # --- Phase 2: CMA-ES ---
        # Parameters for CMA-ES
        # Small population to allow many iterations for early progress
        # For dim=5, standard popsize is 4 + 3*log(5) ~ 8. 
        # Let's try 6 to balance exploration and number of generations.
        popsize = 6 
        mu = popsize // 2  # number of recombination points
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Cumulative path
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Chi_N
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))
        
        # Adaptation parameters
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        mean = best_x.copy()
        
        # Main CMA-ES loop
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Check if we have enough budget for at least 1 generation
            if budget - calls_made < popsize:
                # Not enough for a full population, try to use remaining for single steps?
                # Or just break. Breaking is safer for CMA-ES stability.
                # But we can do a few (1+1) steps to squeeze out more.
                # Let's just break to avoid breaking CMA-ES state.
                break

            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
                
            population = []
            for i in range(popsize):
                z = rng.standard_normal(dim)
                x_i = mean + sigma * (L @ z)
                population.append(x_i)
                
            values = []
            for i in range(popsize):
                if calls_made >= budget or time.time() > deadline:
                    break
                f_i = func(population[i])
                calls_made += 1
                values.append(f_i)
                
            if len(values) < popsize:
                # If we couldn't evaluate the whole population, 
                # we can still use the partial results for a rough update 
                # or just stop. Stopping is safer for CMA-ES stability.
                # However, we can update best if any were better.
                for i, f_i in enumerate(values):
                    if f_i < best_f:
                        best_f = f_i
                        best_x = population[i].copy()
                break

            # Sort by fitness
            indices = np.argsort(values)
            x_sorted = [population[i] for i in indices]
            f_sorted = [values[i] for i in indices]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()
                
            # Update mean
            mean_new = np.zeros(dim)
            for i in range(mu):
                mean_new += weights[i] * x_sorted[i]
                
            # Update ps
            # Compute C_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            z_mean = (mean_new - mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_inv_sqrt @ z_mean)
            
            # Update pc
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2)) < (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - cc) * pc + h_sigma * cc * z_mean
            
            # Update C
            delta_h = (1 - h_sigma) * cc * (2 - cc)
            C_update = c1 * (np.outer(pc, pc) - C)
            for i in range(mu):
                diff = x_sorted[i] - mean
                C_update += (cmu / (sigma**2)) * weights[i] * np.outer(diff, diff)
                
            C = (1 + c1 + cmu - c1 - delta_h) * C + C_update
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2.0
            eigvals_C = np.linalg.eigvalsh(C)
            if np.min(eigvals_C) < 1e-12:
                C += 1e-12 * np.eye(dim)
                
            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            mean = mean_new
            
            # Stopping criteria
            if sigma < 1e-8 or sigma > 1e8:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
