"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES implementation for BBOB f23 (Katsuura)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # CMA-ES Parameters
        # Default population size is 4 + floor(3 * log(dim))
        # For dim=5, popsize = 4 + floor(3 * 1.6) = 4 + 4 = 8.
        # Let's use a slightly larger population for better exploration in early stages, 
        # but keep it small enough to fit many generations in 500 calls.
        # popsize = 10 is a good balance.
        popsize = 10 
        if popsize % 2 == 0:
            popsize += 1  # CMA-ES usually prefers odd popsize? Not strictly, but let's keep it standard.
        
        # Actually, standard CMA-ES popsize is lambda. 
        # Let's stick to lambda = 10 (even is fine, or 11). Let's use 10.
        # Standard weights: w_i = log(lambda/2 + 1) - log(i) for i=1..lambda/2
        # mu = lambda/2 (top half used for update)
        
        lambda_ = popsize
        mu = lambda_ // 2
        weights = np.log(lambda_ / 2 + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Initial parameters
        x_mean = np.zeros(dim)
        sigma = 2.0 # Initial step size. Domain is [-5, 5].
        
        # State variables
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Pre-compute terms
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        hs = (1 + 2 / (dim + 1)) ** (c_sigma / (2 * lambda_))
        
        calls_made = 0
        best_x = x_mean.copy()
        best_f = np.inf
        
        # Helper to evaluate and track best
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Initial evaluation of mean
        best_f = evaluate(x_mean)
        
        while calls_made < budget:
            # Generate population
            # We need to generate lambda_ offspring.
            # If remaining budget < lambda_, we can't do a full generation.
            # We can do partial generation or stop.
            # CMA-ES relies on full generations for updates. 
            # If we can't do a full generation, we might just do random search or stop.
            # Given budget=500 and lambda=10, we have ~50 generations. This is fine.
            
            if budget - calls_made < lambda_:
                # Not enough budget for a full generation.
                # Perform remaining evaluations as random perturbations around best_x
                # or just stop. Stopping is safer to avoid bias.
                # However, "repeating the last n_i" suggests that making fewer calls is okay, 
                # but we want to use the budget.
                # Let's do simple random search with small sigma for remaining budget.
                sigma_local = 0.1
                for _ in range(budget - calls_made):
                    if calls_made >= budget: break
                    y = np.clip(best_x + sigma_local * np.random.randn(dim), -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                break

            # Sample offspring
            offspring = []
            offspring_f = []
            
            # Generate random directions
            # Using standard normal
            rand_vecs = np.random.randn(lambda_, dim)
            
            # Transform to covariance C
            # Efficient way: Cholesky decomposition of C
            # C = L L^T
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes non-positive definite
                C = np.eye(dim)
                L = np.eye(dim)
                
            # y = L @ z
            # z are rand_vecs
            z = rand_vecs
            # x = x_mean + sigma * (L @ z.T).T
            # Actually, standard CMA-ES samples: x_i = x_mean + sigma * (L @ z_i)
            # where z_i ~ N(0, I)
            
            # Compute all offspring
            # z is (lambda, dim). We want (lambda, dim) result.
            # x_i = x_mean + sigma * (L @ z_i.T).T
            # This is equivalent to sigma * (z @ L.T) + x_mean
            # Let's verify: (z @ L.T)[:, i] = sum_k z[k, i] L[k, j] ... wait.
            # Standard: v = L z. v is vector.
            # For matrix Z (lambda x dim), we want V = Z L.T ?
            # If Z is row vectors z_i^T, then z_i L.T gives row vector.
            # So V = Z @ L.T.
            
            V = rand_vecs @ L.T
            offspring = np.array([x_mean + sigma * v for v in V])
            
            # Clip to bounds
            offspring = np.clip(offspring, -5, 5)
            
            # Evaluate offspring
            for i in range(lambda_):
                if calls_made >= budget:
                    break
                fi = evaluate(offspring[i])
                offspring_f.append((fi, i))
            
            if len(offspring_f) < lambda_:
                # Didn't finish generation, break
                break
                
            # Sort offspring by fitness
            offspring_f.sort(key=lambda x: x[0])
            
            # Update mean, sigma, C, pc, ps
            # Top mu individuals
            indices = [idx for _, idx in offspring_f[:mu]]
            weighted_sum = np.zeros(dim)
            for w, idx in zip(weights, indices):
                weighted_sum += w * offspring[idx]
            
            # New mean
            x_mean_new = weighted_sum
            
            # Update path evolution strategies
            # ps = (1 - c_sigma) ps + c_sigma / sigma * sqrt(mu_eff) * (x_mean_new - x_mean)
            # Note: The term (x_mean_new - x_mean) is the step.
            # Standard formula: ps = (1-cs)ps + cs/sigma * sqrt(mu_eff) * (xmean_new - xmean)
            # Wait, usually it's: ps = (1-cs)ps + cs/sigma * sqrt(mu_eff) * (xmean_new - xmean)
            # Let's check standard CMA-ES algorithm.
            # ps = (1 - c_sigma) * ps + c_sigma / sigma * sqrt(mu_eff) * (x_mean_new - x_mean)
            # This assumes the step is normalized by sigma?
            # Actually, the step is x_mean_new - x_mean.
            # The update is: ps = (1 - c_sigma) ps + c_sigma / sigma * sqrt(mu_eff) * (x_mean_new - x_mean)
            
            # Correction: The standard update is
            # ps = (1 - c_sigma) ps + c_sigma / sigma * sqrt(mu_eff) * (x_mean_new - x_mean)
            # This is correct.
            
            ps = (1 - c_sigma) * ps + (c_sigma / sigma) * np.sqrt(mu_eff) * (x_mean_new - x_mean)
            
            # Update pc
            # pc = (1 - c_c) pc + h_c * sqrt(mu_eff) * (x_mean_new - x_mean) / sigma
            # h_c = ||ps|| / chi_n < (1 + 2 / (dim + 1)) ** (c_sigma / (2 * lambda_))
            
            norm_ps = np.linalg.norm(ps)
            h_c = 1.0 if norm_ps / chi_n < (1 + 2 / (dim + 1)) ** (c_sigma / (2 * lambda_)) else 0.0
            
            pc = (1 - c_c) * pc + h_c * np.sqrt(mu_eff) * (x_mean_new - x_mean) / sigma
            
            # Update Sigma
            # sigma = sigma * exp(c_sigma / (2 * lambda_) * (||ps|| / chi_n - 1))
            sigma = sigma * np.exp((c_sigma / (2 * lambda_)) * (norm_ps / chi_n - 1))
            
            # Update Covariance Matrix C
            # C = (1 - c_1 - c_mu) C + c_1 (pc pc^T + (h_c - 1 + c_c) C) + c_mu sum(w_i (y_i y_i^T))
            # where y_i = (x_i - x_mean_old) / sigma
            
            # Calculate y_i for top mu
            y_list = []
            for idx in indices:
                y_i = (offspring[idx] - x_mean) / sigma
                y_list.append(y_i)
            
            # Sum of weighted outer products
            sum_yy = np.zeros((dim, dim))
            for w, y_i in zip(weights, y_list):
                sum_yy += w * np.outer(y_i, y_i)
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (h_c - 1 + c_c) * C) + c_mu * sum_yy
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            # Check eigenvalues? Too expensive. Just proceed.
            
            x_mean = x_mean_new

        return best_x, best_f
# EVOLVE-BLOCK-END
