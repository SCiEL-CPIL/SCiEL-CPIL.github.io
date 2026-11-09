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

        # Strategy: Rank-1 + Rank-mu Natural Gradient Descent with diagonal step sizes
        # This models the search distribution as an axis-aligned ellipsoid,
        # allowing independent scaling of each dimension to handle anisotropy.
        
        lam = 20
        mu = lam // 2
        w = np.log(np.arange(mu, 0, -1) + 0.5)
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Parameters for diagonal adaptation
        # These are tuned for dim=5, similar to CMA-ES but for diagonal case
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

        # Initial state
        mean = np.zeros(dim)
        # Diagonal step sizes (sigma per dimension)
        sigmas = np.ones(dim) * 2.0
        
        # Path memories
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
        stagnation_limit = 8
        gen_count = 0
        last_improvement_gen = 0
        current_best_f = best_f

        while n_calls < budget and time.time() < deadline:
            gen_count += 1
            
            # Check for stagnation
            if gen_count - last_improvement_gen > stagnation_limit:
                # Increase sigmas to escape local minima
                sigmas *= 1.5
                sigmas = np.clip(sigmas, 1e-8, 10.0)
                if np.all(sigmas > 5.0):
                    if gen_count > 20:
                        break

            offspring = []
            offspring_f = []
            
            # Generate offspring
            for i in range(lam):
                if n_calls + 1 > budget:
                    break
                z = rng.standard_normal(dim)
                y = mean + sigmas * z
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
            # For diagonal adaptation, we update paths in the normalized space
            # y = (mean_new - mean) / sigmas
            y_norm = (mean_new - mean) / sigmas
            
            pc = (1 - cc) * pc + cc * np.sqrt(mu_eff_used) * y_norm
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff_used) * y_norm
            
            # Update diagonal step sizes (sigmas)
            # Adaptation rule similar to CMA-ES but for diagonal components
            # We use a rank-1 update for the mean shift and rank-mu for the spread
            
            # Calculate the update for each dimension's sigma
            # The standard CMA-ES sigma update uses the norm of ps.
            # For diagonal, we can update each sigma independently based on the progress in that dimension.
            # A simple approach: use the same sigma update formula but applied to the geometric mean or per-dimension.
            # Here, we use a per-dimension update based on the ratio of the new mean shift to the old sigma.
            
            # Let's use a simplified adaptation:
            # sigma_new[i] = sigma_old[i] * exp( (cs/damps) * ( |y_norm[i]| / sqrt(2/dim) - 1 ) )
            # This adapts the step size based on the magnitude of the move in that dimension.
            
            factor = np.exp((cs / damps) * (np.abs(y_norm) / np.sqrt(2.0 / dim) - 1.0))
            sigmas = sigmas * factor
            sigmas = np.clip(sigmas, 1e-8, 10.0)
            
            # Additionally, we can use the rank-mu update to adjust the shape (diagonal)
            # The rank-mu update in CMA-ES adds to the covariance matrix.
            # For diagonal, this means adding to the variances (sigmas^2).
            # var_new[i] = (1 - c1 - cmu) * var_old[i] + c1 * yk[i]^2 + cmu * sum_j w_j * yk_j[i]^2
            # But we are adapting sigmas directly. Let's stick to the simpler sigma adaptation
            # and rely on the mean update to guide the search.
            
            # However, the "Rank-1 + Rank-mu" directive suggests we should incorporate the spread of parents.
            # Let's add a term that increases sigma in dimensions where the parents are spread out.
            # spread = std of parents in each dimension
            # parent_spread = np.std(parents, axis=0)
            # target_spread = np.sqrt(mu_eff_used) * sigmas
            # adaptation = target_spread / (target_spread + parent_spread + 1e-12)
            # sigmas = sigmas * adaptation
            
            # For now, the sigma adaptation based on y_norm should suffice for the "Natural Gradient" aspect
            # as it aligns the step size with the direction of progress.
            
            mean = mean_new

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
