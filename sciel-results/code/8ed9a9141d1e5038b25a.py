"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size: typically 4 + floor(3 * ln(dim))
        # For dim=5, 4 + 3*1.6 = 8.8 -> 9.
        # Let's use a slightly larger population for robustness in multimodal cases, 
        # but keep it small enough to fit in budget 500.
        # 500 / 9 = 55 generations.
        # 500 / 12 = 41 generations.
        # Let's stick to standard recommendation: lambda = 4 + floor(3 * ln(dim))
        # dim=5 -> ln(5)=1.609, 3*1.609=4.8, floor=4, lambda=8.
        # Let's use lambda=10 for slightly better exploration.
        lambda_pop = 10
        mu = lambda_pop // 2  # 5
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = (weights.sum() ** 2) / np.sum(weights ** 2)
        
        # Learning rates
        # Standard formulas
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Initial step size
        # The search space is [-5, 5]^5.
        # Büche-Rastrigin has a global minimum that can be shifted.
        # A reasonable initial sigma is 1.0 or 2.0.
        # If we start at 0, we might be far from the min if shifted.
        # Let's start at 0.0 and let CMA-ES adapt.
        # Or maybe a random start? CMA-ES is usually robust to start point.
        # Given the hidden shift, starting at 0 is a decent guess if shift is small,
        # but if shift is large, we need to move.
        # Let's start at 0.
        x_mean = np.zeros(dim)
        sigma = 1.0
        
        # Initial covariance matrix: Identity
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Cholesky decomposition of C
        # We need to update Cholesky factor after each generation
        # For simplicity and speed in Python, we might recompute or use update formula.
        # Recomputing Cholesky of 5x5 is very fast.
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1
        
        if calls >= budget:
            return best_x, best_f
            
        # Pre-compute normal matrix? No, C changes.
        
        # Main loop
        while calls < budget:
            if time.time() >= deadline:
                break
                
            # Check if we have enough budget for a full generation
            if calls + lambda_pop > budget:
                # Do a partial generation or break?
                # CMA-ES works best with full generations.
                # If we can't do a full one, we might do a few single steps (1+1-ES style)
                # or just break.
                # Let's break to avoid wasting time on a broken population update.
                # Or we can run (1+1)-ES for the remaining calls.
                # Given budget 500 and lambda 10, we have 50 full generations.
                # It's unlikely to end mid-generation unless budget isn't multiple of lambda.
                # 500 % 10 == 0. So it should be fine.
                break
                
            # Generate population
            population = []
            f_values = []
            
            for i in range(lambda_pop):
                # Sample from N(0, C)
                # y = B * z, where B is Cholesky factor of C
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset to identity
                    C = np.eye(dim)
                    B = np.eye(dim)
                    
                z = np.random.standard_normal(dim)
                y = B @ z
                x_new = x_mean + sigma * y
                
                # Clip to bounds
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                calls += 1
                
                population.append(x_new)
                f_values.append(f_new)
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    
                # Early break if time is tight
                if time.time() >= deadline - 1.0:
                    # We still need to finish the generation for valid CMA-ES update?
                    # If we break here, the population is incomplete.
                    # It's safer to finish the generation if we have time.
                    # If we don't have time, break out of the whole loop.
                    # Let's assume we have enough time for the rest of the generation.
                    pass

            # Sort population by fitness
            indices = np.argsort(f_values)
            population = [population[i] for i in indices]
            f_values = [f_values[i] for i in indices]
            
            # Update mean
            # x_mean = sum(weights[i] * population[i])
            x_mean_new = np.zeros(dim)
            for i in range(mu):
                x_mean_new += weights[i] * population[i]
            
            # Update evolution paths
            # p_sigma = (1-cs)*p_sigma + sqrt(cs*(2-cs)*mu_eff) * (1/sigma) * B^T * (x_mean_new - x_mean)/sigma
            # Note: The formula usually involves the normalized step.
            # Let step = (x_mean_new - x_mean) / sigma
            step = (x_mean_new - x_mean) / sigma
            
            # B is Cholesky of C. We need B^T? 
            # The transformation is y = B z. The inverse is B^{-1} y.
            # The standard CMA-ES update for p_sigma uses the transformed space.
            # Actually, the path is in the "normal" space or the "adapted" space?
            # Standard implementation:
            # p_sigma = (1-cs)*p_sigma + sqrt(cs*(2-cs)*mu_eff) * (1/sigma) * B^{-1} * (x_mean_new - x_mean)
            
            # Calculate B^{-1}
            # Since B is lower triangular, we can solve B v = step
            try:
                B_inv_step = np.linalg.solve(B, step)
            except np.linalg.LinAlgError:
                B_inv_step = step # Fallback
            
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * B_inv_step
            
            # Check h_sigma
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-cs)^{2(gen+1)}) < (1.4 + 2/(dim+1)) * sqrt((mu_eff+2)-2)
            # We need to track generation count for the denominator
            # Let's simplify: use a fixed threshold or approximate.
            # For robustness, we can just use the norm check against a constant.
            # ||p_sigma|| / sqrt(1 - (1-cs)^{2}) is the standard form for first gen.
            # Let's assume we are past the initial transient.
            h_sigma = 1.0
            # More accurate:
            # We need to know the number of generations.
            # Let's just use the condition:
            # if ||p_sigma|| / sqrt(1 - (1-cs)**2) < (1.4 + 2/(dim+1)) * sqrt((mu_eff+2)-2):
            #    h_sigma = 1
            # else:
            #    h_sigma = 0
            # This is an approximation but often works.
            
            # Update p_c
            # p_c = (1-cc)*p_c + h_sigma * sqrt(cc*(2-cc)*mu_eff) * B^{-1} * (x_mean_new - x_mean)
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * B_inv_step
            
            # Update Covariance Matrix C
            # C = (1 - c1 - cmu) * C + c1 * (p_c * p_c^T + (1-h_sigma)*cc*(2-cc)*C) + cmu * sum(weights[i] * (y_i * y_i^T))
            # where y_i = (x_i - x_mean_old) / sigma
            
            # Calculate y_i for top mu individuals
            outer_products = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (population[i] - x_mean) / sigma
                outer_products += weights[i] * np.outer(y_i, y_i)
            
            # Rank-one update
            rank_one = np.outer(p_c, p_c)
            if h_sigma == 0:
                rank_one += (1 - h_sigma) * cc * (2 - cc) * C
            
            C = (1 - c1 - cmu) * C + c1 * rank_one + cmu * outer_products
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = x_mean_new
            
            # Update sigma
            # sigma = sigma * exp(cs/chi0 * (||p_sigma||/chi0 - 1))
            # chi0 = sqrt(dim) * (1 - 1/(2*dim^2) - 1/(8*dim^4))
            chi0 = np.sqrt(dim) * (1 - 1/(2*dim**2) - 1/(8*dim**4))
            sigma = sigma * np.exp(cs / chi0 * (np.linalg.norm(p_sigma) / chi0 - 1))
            
            # Prevent sigma from becoming too large or small
            if sigma > 10.0:
                sigma = 10.0
            if sigma < 1e-6:
                sigma = 1e-6
                # If sigma is too small, we might be stuck.
                # We could restart here, but CMA-ES usually handles this.
                # If stuck, break to save time?
                # Let's break if no progress in last 20 generations?
                # We don't track progress easily here without extra storage.
                # Let's just continue.

        return best_x, best_f
# EVOLVE-BLOCK-END
