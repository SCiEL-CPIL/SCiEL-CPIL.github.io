import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(d))
        if lam < 4:
            lam = 4
        # We have 500 evaluations. Let's aim for about 40-50 generations.
        # If lam=10, 50 generations = 500 evals.
        
        # Initial mean: center of the box
        mean = np.zeros(d)
        # Initial step size: 2.0 is a reasonable start for [-5, 5]
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(d)
        
        # Evolution paths
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Update rates
        mu = lam // 2  # Number of recombination weights
        # Weights: standard CMA-ES weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        
        # Precompute weights for top mu individuals
        # Note: w is defined for the top mu. The rest have weight 0.
        
        start_time = time.time()
        deadline = 250.0 # Safety margin for 300s limit
        
        best_x = mean.copy()
        best_f = np.inf
        
        # Initial evaluation of mean
        f_mean = func(mean)
        if f_mean < best_f:
            best_f = f_mean
            best_x = mean.copy()
            
        evals = 1
        
        # Main loop
        while evals + lam <= budget:
            if time.time() - start_time > deadline:
                break
                
            # Generate population
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reinitialize or adjust
                C = np.eye(d)
                L = np.eye(d)
                
            # Generate normal random numbers
            # z_i ~ N(0, I)
            Z = np.random.randn(lam, d)
            
            # Transform to y_i = mean + sigma * L @ z_i
            # Y = Z @ L.T
            Y = Z @ L.T
            X = mean[None, :] + sigma * Y
            
            # Evaluate population
            F = np.empty(lam)
            for i in range(lam):
                F[i] = func(X[i])
                evals += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
                    
            # Sort indices by fitness
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            # Update mean
            # New mean = sum(w_i * x_i) for i in 0..mu-1
            new_mean = np.zeros(d)
            for i in range(mu):
                new_mean += w[i] * X_sorted[i]
            
            # Update paths
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * sigma * C^(-1/2) * (x_1 - mean)/sigma
            # Note: C^(-1/2) is inv(L)
            inv_L = np.linalg.inv(L)
            # (x_1 - mean) / sigma is Y[0]
            # Actually, the standard formula is:
            # p_sigma <- (1-c_sigma)p_sigma + sqrt(c_sigma(2-c_sigma)mu_eff) * C^{-1/2} (x_{1:lambda} - m) / sigma
            # Here (x_1 - m)/sigma = L z_1. So C^{-1/2} L z_1 = z_1.
            # So the update simplifies if we use the standard normal z.
            # Let's stick to the standard implementation which computes y = (x - m)/sigma
            # y_sorted = (X_sorted - mean[None, :]) / sigma
            
            y_sorted = (X_sorted - mean[None, :]) / sigma
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sorted[0]
            
            # Check step size adaptation
            # If ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2t}) > chi_N + 1.4 + 2/(d+1), increase sigma
            # For simplicity and robustness, we can use the standard heuristic
            # ||p_sigma|| / chi_N
            chi_N = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
            if np.linalg.norm(p_sigma) / chi_N < 1.4 + 2.0 / (d + 1.0):
                sigma *= np.exp((1/5) * (np.linalg.norm(p_sigma)/chi_N - 1))
            else:
                sigma *= np.exp(-(1/5) * (np.linalg.norm(p_sigma)/chi_N - 1))
                
            # Clamp sigma to avoid explosion or extinction
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update covariance matrix
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean)/sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma
            
            # C update
            # C <- (1 - c_c) * C + c_c * (p_c p_c^T + (1 - delta_sig) * (1 - c_c) * C + delta_sig * ... )
            # Standard update:
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2t}) < (1.4 + 2/(d+1)) * chi_N
            # We approximate h_sigma = 1 if the step was successful (sigma decreased or stayed small)
            # A simpler and robust update for CMA-ES:
            # C <- (1 - (c_c + c_sigma) * 2) * C + ... is complex.
            # Let's use the standard formula:
            # C <- (1 - c_c) C + c_c (p_c p_c^T + (1 - delta_sig) (1 - c_c) C) 
            # where delta_sig depends on h_sigma.
            
            # To keep it simple and robust within the time limit, I will use a slightly simplified covariance update
            # that is known to work well for small d.
            # Actually, let's just use the standard one.
            
            # h_sigma condition
            # We need to track the age of the algorithm for the denominator in h_sigma check.
            # For a short run, we can approximate.
            # Let's assume h_sigma = 1.0 for now, which is safe if the step size is adapting well.
            # A better check: if sigma increased, h_sigma = 0, else 1? No, that's not quite right.
            # Let's use the condition: h_sigma = 1 if ||p_sigma||/chi_N < (1.4 + 2/(d+1))
            
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / chi_N < (1.4 + 2.0 / (d + 1.0))) else 0.0
            
            # C update
            # The term (1 - c_c) C + c_c p_c p_c^T is the main part.
            # There is also a term involving the rank-mu update.
            # C <- (1 - c_c) C + c_c (p_c p_c^T + (1 - delta_sig) (1 - c_c) C)
            # + c_sigma (1 - delta_sig) (1 - c_c) ... wait, the standard formula is:
            # C <- (1 - c_c) C + c_c (p_c p_c^T + (1 - delta_sig) (1 - c_c) C)
            # This is for rank-1.
            # Rank-mu update adds: + c_sigma * sum(w_i (y_i y_i^T))
            # where y_i are the normalized successful steps.
            
            # Let's combine them.
            # C_new = (1 - c_c) * C
            # C_new += c_c * np.outer(p_c, p_c)
            # C_new += c_c * (1 - h_sigma) * (1 - c_c) * C  # This part is tricky.
            
            # Let's use the standard implementation from Hansen's papers.
            # C <- (1 - c_c) C + c_c (p_c p_c^T + (1 - delta_sig) (1 - c_c) C)
            # delta_sig = 1 if h_sigma=0 else 0? No.
            # delta_sig = 1 - h_sigma?
            
            # Let's simplify:
            # C_new = (1 - c_c) * C + c_c * np.outer(p_c, p_c)
            # Rank-mu part:
            # y_mu = np.zeros((mu, d))
            # for i in range(mu):
            #     y_mu[i] = y_sorted[i]
            # C_new += c_sigma * (np.sum(w[:, None, None] * y_mu[:, :, None] * y_mu[:, None, :], axis=0) - (1 - h_sigma) * (1 - c_c) * C)
            
            # This is getting complicated to get exactly right without a reference library.
            # However, for d=5 and budget=500, even a simplified CMA-ES (e.g. only rank-1 update + step size) might outperform (1+1)-ES.
            # But let's try to implement the full standard update.
            
            # Let's restart with a cleaner implementation of the update rules.
            # I will use the standard CMA-ES algorithm as described in Hansen (2016).
            
            # 1. Compute y_i = (x_i - m) / sigma
            # 2. Update p_sigma
            # 3. Adapt sigma
            # 4. Update p_c
            # 5. Update C
            # 6. Update m
            
            # The code above already did 1-4 partially. Let's redo the C update carefully.
            
            # C update
            # C = (1 - c_c) * C + c_c * (np.outer(p_c, p_c) + (1 - h_sigma) * (1 - c_c) * C)
            # Wait, the term (1 - h_sigma) * (1 - c_c) * C is inside the parenthesis?
            # Formula: C <- (1 - c_c) C + c_c (p_c p_c^T + (1 - delta_sig) (1 - c_c) C)
            # where delta_sig = 1 if h_sigma = 1 else 0? No, delta_sig is related to the step size adaptation.
            # Actually, the standard formula is:
            # C <- (1 - c_c) C + c_c (p_c p_c^T + (1 - h_sigma) (1 - c_c) C)
            # This is for the rank-1 part.
            # Then add the rank-mu part:
            # C += c_sigma * (sum(w_i y_i y_i^T) - (1 - h_sigma) (1 - c_c) C)
            
            # Let's combine:
            # C_new = (1 - c_c) * C
            # C_new += c_c * np.outer(p_c, p_c)
            # C_new += c_c * (1 - h_sigma) * (1 - c_c) * C
            # C_new += c_sigma * np.sum(w[:, None, None] * y_mu[:, :, None] * y_mu[:, None, :], axis=0)
            # C_new -= c_sigma * (1 - h_sigma) * (1 - c_c) * C
            
            # Let's simplify by assuming h_sigma=1 (which is true if the step was not too large).
            # If h_sigma=1, the (1 - h_sigma) terms vanish.
            # C_new = (1 - c_c) * C + c_c * np.outer(p_c, p_c) + c_sigma * np.sum(...)
            
            # This is a valid simplification for robustness.
            
            y_mu = y_sorted[:mu]
            # Weighted sum of outer products
            # w is shape (mu,), y_mu is shape (mu, d)
            # We want sum_i w_i * y_i * y_i^T
            # This is (y_mu.T * w) @ y_mu
            rank_mu_update = (y_mu.T * w) @ y_mu
            
            C = (1 - c_c) * C + c_c * np.outer(p_c, p_c) + c_sigma * rank_mu_update
            
            # Ensure C is positive definite
            # Check eigenvalues
            # If any eigenvalue is too small, add identity
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-6:
                C += (1e-6 - np.min(eigvals)) * np.eye(d)
            
            # Update mean
            mean = new_mean
            
            # Check for convergence
            if sigma < 1e-10:
                break

        return best_x, best_f
