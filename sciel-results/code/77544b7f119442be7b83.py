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

        # CMA-ES Parameters
        # For dim=5, budget=500.
        # A single run with lambda=20 allows for 25 generations.
        # This provides enough data points per generation to estimate covariance reliably.
        lam = 20
        mu = lam // 2
        w = np.log(np.arange(mu, 0, -1) + 0.5)
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

        # Initial state
        # Start at the center, which is a reasonable prior for BBOB functions
        # unless the shift is extreme, but the search should find the basin quickly.
        mean = np.zeros(dim)
        # Initial sigma: 2.0 covers a large portion of [-5,5]^5
        sigma = 2.0
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Evaluate initial mean
        f_mean = call_f(mean)
        if f_mean is None:
            return best_x, best_f
        if f_mean < best_f:
            best_f = f_mean
            best_x = mean.copy()

        # Main Loop
        # We run until budget is exhausted or we detect stagnation.
        # Stagnation is defined as no improvement in best_f for 'stagnation_limit' generations.
        stagnation_limit = 8
        gen_count = 0
        last_improvement_gen = 0
        current_best_f = best_f

        while n_calls < budget and time.time() < deadline:
            gen_count += 1
            
            # Check for stagnation
            if gen_count - last_improvement_gen > stagnation_limit:
                # If we have used more than 50% of the budget and are stuck,
                # we might try a small perturbation of the mean to escape, 
                # or just break if we are very close to the boundary of convergence.
                # For a "Discus" function, getting stuck usually means we are in a local basin.
                # A simple restart is risky with limited budget. 
                # Instead, we increase sigma slightly to allow for a larger jump.
                sigma *= 1.5
                if sigma > 5.0:
                    # If sigma is already large and we are stuck, we are likely stuck in a bad spot.
                    # Try a random restart from a new point? 
                    # With budget 500, we have limited room for restarts.
                    # Let's just break if we've been stuck for a long time and sigma is large.
                    if gen_count > 20:
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
            
            # Check for improvement
            if offspring_f[0] < current_best_f:
                current_best_f = offspring_f[0]
                last_improvement_gen = gen_count

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
            
            # Regularize C to ensure positive definiteness
            C += 1e-12 * np.eye(dim)
            
            # Compute inverse square root for next iteration
            try:
                U, s, Vt = np.linalg.svd(C)
                s_inv_sqrt = 1.0 / np.sqrt(s + 1e-12)
                C_inv_sqrt = U @ np.diag(s_inv_sqrt) @ Vt
            except np.linalg.LinAlgError:
                # Fallback if SVD fails (unlikely with regularization)
                C_inv_sqrt = np.eye(dim)
                
            mean = mean_new
            
            # Adapt sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # If sigma becomes very small, we have likely converged locally.
            if sigma < 0.01:
                # Try a small restart? Or just break?
                # Given the budget, breaking is safer to preserve the best found.
                # But we might have budget left. Let's do a small polish.
                break

        # Final Polish with (1+1)-ES on the global best if budget remains
        remaining = budget - n_calls
        if remaining > 10 and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            sigma_poly = 0.1
            # Use remaining budget for local refinement
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
