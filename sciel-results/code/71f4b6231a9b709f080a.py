"""
Strategy: Parallel Multi-Start CMA-ES with Adaptive Restarting.
Rationale: BBOB f11 (Discus) has a complex landscape with potential local optima.
A single CMA-ES from the center may fail. Multiple parallel searches increase coverage.
We use small populations (lambda=8) to allow more restarts and faster convergence per instance.
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
        rng = np.random.default_rng(12345)
        
        best_x_global = None
        best_f_global = float('inf')
        
        def call_f(x):
            nonlocal n_calls
            if n_calls >= budget:
                return None
            n_calls += 1
            return func(x)

        def update_global(x, f):
            nonlocal best_x_global, best_f_global
            if f < best_f_global:
                best_x_global = x.copy()
                best_f_global = f

        # CMA-ES implementation for a single instance
        def run_cma_es(initial_mean, initial_sigma, max_calls, rng_instance):
            # Parameters for dim=5
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
            
            mean = initial_mean.copy()
            sigma = initial_sigma
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            local_best_x = mean.copy()
            local_best_f = float('inf')
            calls_used = 0
            
            # Initial evaluation
            f_mean = call_f(mean)
            if f_mean is None:
                return local_best_x, local_best_f, calls_used
            calls_used += 1
            local_best_f = f_mean
            update_global(mean, f_mean)
            
            stagnation_counter = 0
            max_stagnation = 5 # Generations without improvement before restart signal
            
            while calls_used < max_calls and n_calls < budget:
                if time.time() > deadline - 5:
                    break
                
                # Generate offspring
                offspring = []
                offspring_f = []
                
                for i in range(lam):
                    if calls_used + 1 > max_calls or n_calls >= budget:
                        break
                    z = rng_instance.standard_normal(dim)
                    y = mean + sigma * (C_inv_sqrt @ z)
                    y = np.clip(y, -5.0, 5.0)
                    
                    fy = call_f(y)
                    if fy is None:
                        break
                    calls_used += 1
                    offspring.append(y)
                    offspring_f.append(fy)
                
                if not offspring:
                    break
                
                offspring = np.array(offspring)
                offspring_f = np.array(offspring_f)
                
                # Sort
                idx = np.argsort(offspring_f)
                offspring = offspring[idx]
                offspring_f = offspring_f[idx]
                
                # Update local best
                if offspring_f[0] < local_best_f:
                    local_best_x = offspring[0].copy()
                    local_best_f = offspring_f[0]
                    update_global(local_best_x, local_best_f)
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    
                if stagnation_counter > max_stagnation:
                    break # Signal restart
                
                # Selection
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
                
                # Path evolution
                pc = (1 - cc) * pc + cc * np.sqrt(mu_eff_used) * (C_inv_sqrt @ (mean_new - mean) / sigma)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff_used) * (C_inv_sqrt @ (mean_new - mean) / sigma)
                
                # Covariance update
                yk = (mean_new - mean) / sigma
                C = (1 + c1 * (2 * cc - 1) - c1 * cmu) * C + c1 * np.outer(yk, yk)
                
                yk_parents = (parents - mean) / sigma
                C = C + cmu * np.sum(w_used[:, np.newaxis, np.newaxis] * np.einsum('i,ij,ik->ijk', w_used, yk_parents, yk_parents), axis=0)
                
                C = (C + C.T) / 2.0
                
                # Compute C_inv_sqrt
                U, s, Vt = np.linalg.svd(C)
                s_inv_sqrt = 1.0 / np.sqrt(s + 1e-12)
                C_inv_sqrt = U @ np.diag(s_inv_sqrt) @ Vt
                
                mean = mean_new
                
                # Adapt sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
                sigma = np.clip(sigma, 1e-8, 50.0)
                
                if sigma < 0.01:
                    break # Convergence
                    
            return local_best_x, local_best_f, calls_used

        # Multi-Start Strategy
        # We divide the budget among K parallel "slots".
        # Since we run sequentially in Python, we simulate parallelism by time-slicing or call-slicing.
        # Given the short budget (500), we will run a sequence of short CMA-ES instances.
        # Start with a few diverse points.
        
        # Initial diverse set of starting points
        start_points = []
        # Center
        start_points.append(np.zeros(dim))
        # Random points in [-4, 4]^5
        for _ in range(9):
            start_points.append(rng.uniform(-4, 4, size=dim))
            
        # Shuffle to ensure diversity in order
        rng.shuffle(start_points)
        
        # We will try to run each instance with a budget of ~50 calls.
        # If an instance finishes early (converged/stagnant), we start the next one immediately.
        # If it uses all 50, we move to the next.
        # We have 10 instances * 50 calls = 500 calls.
        
        current_idx = 0
        instance_budget = 50
        
        while n_calls < budget and time.time() < deadline - 10:
            if current_idx >= len(start_points):
                # If we run out of pre-defined points, generate new random ones
                # Or restart from best point found so far with larger sigma?
                # Let's just generate new random points
                start_points.append(rng.uniform(-4, 4, size=dim))
                
            init_mean = start_points[current_idx]
            current_idx += 1
            
            # Initial sigma: slightly smaller than domain size to be local, but large enough to explore
            # If we are restarting a stagnant one, we might want to jump.
            # For simplicity, use a fixed moderate sigma.
            init_sigma = 1.5 
            
            # Run CMA-ES with limited budget
            bx, bf, used = run_cma_es(init_mean, init_sigma, instance_budget, rng)
            
            # If it used all budget, we are done with this "slot"
            # If it stopped early, we can start a new one immediately.
            # This effectively creates a dynamic allocation.
            
        # If we still have budget left (e.g., due to time check or early breaks),
        # do a final polish on the global best
        remaining = budget - n_calls
        if remaining > 5 and best_x_global is not None and time.time() < deadline - 5:
            x = best_x_global.copy()
            fx = best_f_global
            sigma_poly = 0.1
            for _ in range(remaining):
                if time.time() > deadline - 2:
                    break
                y = np.clip(x + sigma_poly * rng.standard_normal(dim), -5.0, 5.0)
                fy = call_f(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma_poly *= 1.1
                    if fx < best_f_global:
                        best_x_global = x.copy()
                        best_f_global = fx
                else:
                    sigma_poly *= 0.9
                    sigma_poly = max(sigma_poly, 1e-4)

        if best_x_global is None:
            # Fallback if no calls were made (should not happen)
            best_x_global = np.zeros(dim)
            
        return best_x_global, best_f_global
