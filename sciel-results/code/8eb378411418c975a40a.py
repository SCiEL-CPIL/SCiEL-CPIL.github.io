import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin before 300s kill

        # CMA-ES Parameters
        # Standard lambda for n=5 is ~8. Let's use 10 for slightly better exploration.
        lambda_ = 10 
        # Initial mean: center of the box
        mean = np.zeros(dim)
        # Initial sigma: The box is [-5, 5]^5, width 10.
        # The minimum is in [-4, 4]^5.
        # A sigma of 1.5 allows coverage of most of the interesting region in a few steps.
        sigma = 1.5
        
        # Covariance matrix starts as identity
        C = np.eye(dim)
        # Inverse square root of C (initially identity)
        C_sqrt_inv = np.eye(dim)
        
        # Adaptation parameters
        # Standard CMA-ES parameters
        mu = lambda_ // 2  # Number of weights
        # Weights: rank-mu update
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        # Variance-effectiveness
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        # Normalization factor for ps
        norm_ps = np.sqrt(1 - np.exp(-2 / cs))
        # Normalization factor for pc
        norm_pc = np.sqrt(1 - np.exp(-2 / cs))
        
        # Initial best
        best_x = mean.copy()
        best_f = float('inf')
        
        calls = 0

        # Helper to evaluate a point and track best
        def evaluate(x):
            nonlocal best_x, best_f, calls
            # Clip to box to ensure validity, though CMA-ES usually stays near mean
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Initial evaluation of mean (optional but helps AOCC if mean is good)
        # Actually, let's just run the first generation.
        
        while calls < budget:
            # Time check
            if time.time() > deadline:
                break
            
            # Generate lambda offspring
            # B = C_sqrt_inv @ Random Normal
            # We need to generate random samples from N(0, C)
            # x = mean + sigma * C_sqrt_inv @ z
            
            # Generate random z
            z = np.random.randn(lambda_, dim)
            
            # Transform z
            # x_i = mean + sigma * (C_sqrt_inv @ z_i)
            # Note: C_sqrt_inv is inverse square root of C
            # In standard CMA-ES, we update C and keep C_sqrt_inv approximated
            # For simplicity and robustness in a short budget, we can use the
            # standard update where we maintain C and compute Cholesky or
            # use the fact that C starts as I.
            
            # Efficient implementation:
            # x = mean + sigma * (B @ z) where B is matrix square root of C
            # We maintain C and compute its square root or inverse square root periodically.
            # For n=5, computing sqrtm or cholesky is fast.
            
            # Let's use Cholesky decomposition for stability
            try:
                L = np.linalg.cholesky(C)
                # x = mean + sigma * (L @ z)
                # z is (lambda, dim), L is (dim, dim)
                # We need L @ z.T -> (dim, lambda) -> T -> (lambda, dim)
                x_offspring = mean[None, :] + sigma * (z @ L.T)
            except np.linalg.LinAlgError:
                # Fallback if C becomes singular (unlikely with proper updates)
                x_offspring = mean[None, :] + sigma * z

            # Evaluate offspring
            f_offspring = []
            for i in range(lambda_):
                if calls >= budget:
                    break
                f_val = evaluate(x_offspring[i])
                f_offspring.append(f_val)
                
            # If we ran out of budget during evaluation, break
            if calls >= budget:
                break
            
            # Ensure we have enough evaluations
            if len(f_offspring) < mu:
                break

            # Sort by function value
            # We only care about the best mu
            idx = np.argsort(f_offspring)[:mu]
            y_selected = x_offspring[idx]
            f_selected = np.array(f_offspring)[idx]
            
            # Weighted mean of selected
            y_w = np.sum(weights[:, None] * y_selected, axis=0)
            
            # Update evolution path ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (C_sqrt_inv @ y_w)
            
            # Update evolution path pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls // lambda_))) <
                    (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
            
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * (C_sqrt_inv @ y_w)
            
            # Update covariance matrix C
            # Rank-mu update
            # Delta C = c1 * (pc @ pc.T + (1-hsig)*cs*(2-cs)*C) + cmu * sum(wi * (yi @ yi.T))
            
            # Correction term for pc
            hsig_correction = 1 - hsig
            rank1_update = pc @ pc.T + hsig_correction * cs * (2 - cs) * C
            
            rank_mu_update = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu_update += weights[i] * np.outer(y_selected[i], y_selected[i])
                
            C = (1 + c1 * hsig_correction * cs * (2 - cs) + cmu) * C - c1 * rank1_update - cmu * rank_mu_update
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(cs / (2 * dim) * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls // lambda_))) - 1))
            
            # Update mean
            mean += c1 * (y_w - mean) # Wait, standard is mean = mean + c1 * (weighted_mean - mean)
            # Actually, the standard update is:
            # mean = mean + c1 * (y_w - mean) is incorrect.
            # The mean is updated by the weighted mean of the selected points.
            # In CMA-ES, the mean is the weighted mean of the selected points.
            # So: mean = y_w
            
            mean = y_w
            
            # Update C_sqrt_inv
            # We need to update the inverse square root of C.
            # This is computationally expensive. For n=5, we can just recompute it.
            # C_sqrt_inv = inv(sqrt(C))
            # Using eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Check convergence
            if sigma < 1e-8:
                break

        return best_x, best_f
