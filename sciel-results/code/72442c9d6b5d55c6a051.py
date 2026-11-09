"""
Strategy: Multi-Start Adaptive CMA-ES for BBOB f11 (Discus).
Rationale: The Discus function has a complex landscape with multiple local optima.
A single CMA-ES run, even with a large population, might converge to a local minimum
or waste budget exploring a large area without finding the global basin.
By running multiple smaller CMA-ES instances in parallel (sharing the budget),
we increase the probability of finding the global minimum early (boosting AOCC)
and provide redundancy against local minima.
"""

import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')

        def call_f(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None
            n_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def compute_c_inv_sqrt(C_mat):
            # Use SVD for stability
            U, s, Vt = np.linalg.svd(C_mat)
            s_inv_sqrt = 1.0 / np.sqrt(s + 1e-12)
            return U @ np.diag(s_inv_sqrt) @ Vt

        # Multi-Start Configuration
        # For dim=5, budget=500.
        # Let's use 5 starts. Each gets ~100 calls.
        # Small population lambda=8 allows for ~12 generations per start.
        n_starts = 5
        budget_per_start = budget // n_starts
        
        # CMA-ES parameters for small population
        lam = 8
        mu = lam // 2
        w = np.log(np.arange(mu, 0, -1) + 0.5)
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

        # Initial points: Spread them out to cover the space
        # Center is 0, but let's try some shifted points to escape local minima
        initial_centers = [
            np.zeros(dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-2, 2, dim),
            np.full(dim, 1.0),
            np.full(dim, -1.0)
        ]

        for s in range(n_starts):
            if n_calls >= budget or time.time() > deadline:
                break
                
            # Initialize CMA-ES state for this start
            mean = initial_centers[s % len(initial_centers)].copy()
            # Initial sigma: 1.5 is a good balance for 5D box
            sigma = 1.5
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Evaluate initial mean
            f_mean = call_f(mean)
            if f_mean is None:
                break

            # Run CMA-ES for this start
            # Limit generations to ensure we don't consume too much of the budget
            max_gens = budget_per_start // lam + 1
            for gen in range(max_gens):
                if n_calls >= budget or time.time() > deadline:
                    break
                
                offspring = []
                offspring_f = []
                
                # Generate offspring
                for i in range(lam):
                    if n_calls + 1 > budget:
                        break
                    z = rng.standard_normal(dim)
                    y = mean + sigma * (C_inv_sqrt @ z)
                    y = np.clip(y, -5.0, 5.0)
                    fy = call_f(y)
                    if fy is None:
                        break
                    offspring.append(y)
                    offspring_f.append(fy)
                
                if len(offspring) < 2:
                    break
                
                offspring = np.array(offspring)
                offspring_f = np.array(offspring_f)
                
                # Sort
                idx = np.argsort(offspring_f)
                offspring = offspring[idx]
                offspring_f = offspring_f[idx]
                
                # Select parents
                mu_used = min(mu, len(offspring))
                parents = offspring[:mu_used]
                
                if mu_used < mu:
                    w_used = w[:mu_used]
                    w_used = w_used / w_used.sum()
                    mu_eff_used = 1.0 / np.sum(w_used**2)
                else:
                    w_used = w
                    mu_eff_used = mu_eff
                
                # Update mean
                mean_new = np.sum(w_used[:, np.newaxis] * parents, axis=0)
                
                # Update paths
                pc = (1 - cc) * pc + cc * np.sqrt(mu_eff_used) * (C_inv_sqrt @ (mean_new - mean) / sigma)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff_used) * (C_inv_sqrt @ (mean_new - mean) / sigma)
                
                # Update covariance
                ccs = cs
                norm_ps = np.linalg.norm(ps)
                h_sigma = 1.0 if norm_ps < (1.4 + 2.0 / (dim + 1.0)) * ccs else 0.0
                
                yk = (mean_new - mean) / sigma
                C = (1 + c1 * h_sigma * cc * (cc - 1) - c1 * cmu) * C + c1 * np.outer(yk, yk)
                
                yk_parents = (parents - mean) / sigma
                C = C + cmu * np.sum(w_used[:, np.newaxis, np.newaxis] * np.einsum('i,ij,ik->ijk', w_used, yk_parents, yk_parents), axis=0)
                
                C = (C + C.T) / 2.0
                C_inv_sqrt = compute_c_inv_sqrt(C)
                
                mean = mean_new
                
                # Adapt sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
                sigma = np.clip(sigma, 1e-8, 50.0)
                
                # Stagnation check: if sigma is very small, stop this start
                if sigma < 0.01:
                    break

        # Final polish with (1+1)-ES on the global best
        remaining = budget - n_calls
        if remaining > 5 and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            sigma_poly = 0.1
            for _ in range(remaining):
                if time.time() > deadline - 1:
                    break
                y = np.clip(x + sigma_poly * rng.standard_normal(dim), -5.0, 5.0)
                fy = call_f(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma_poly *= 1.1
                    if fx < best_f:
                        best_x = x.copy()
                        best_f = fx
                else:
                    sigma_poly *= 0.9
                    sigma_poly = max(sigma_poly, 1e-4)

        return best_x, best_f
