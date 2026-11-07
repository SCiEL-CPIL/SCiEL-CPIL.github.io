import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        N = self.dim
        bounds_low = -5.0
        bounds_high = 5.0
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(N))
        if lam < 4: lam = 4
        # For N=5, lam = 4 + 3*1.6 = 8.8 -> 8. Let's use 10 for robustness.
        lam = 10 
        
        mu = lam // 2
        # Weights for recombination
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Learning rates
        cc = (4.0 + mueff / N) / (N + 4.0 + 2.0 * mueff / N)
        cs = (mueff + 2.0) / (N + mueff + 5.0)
        c1 = 2.0 / ((N + 1.3)**2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 2.0 / mueff) / ((N + 2.0)**2 + mueff))
        
        # Initial step size
        # Heuristic: start with a step size that covers a portion of the box
        sigma = 2.0
        
        # Initial mean: random or center? Center is often good if the landscape is not highly skewed.
        # However, the problem statement says "hidden seeded shift... minimizer lies in [-4, 4]^5".
        # A random start within the box is safer to avoid bias if the shift is large.
        # But CMA-ES is good at moving. Let's start at 0.0 vector, or a random point.
        # Given the box is [-5, 5] and minimizer in [-4, 4], 0 is a reasonable guess.
        # To be safe against shifts, let's use a small random perturbation or just 0.
        # Let's use a random point from a uniform distribution to be unbiased.
        rng = np.random.default_rng(42)
        mean = rng.uniform(-5.0, 5.0, N)
        
        # Covariance matrix
        C = np.eye(N)
        # Square root of C for sampling
        C_sqrt = np.eye(N)
        
        # Evolution paths
        p_sigma = np.zeros(N)
        p_c = np.zeros(N)
        
        # Track best
        best_x = mean.copy()
        best_f = np.inf
        calls = 0
        
        # Pre-compute Cholesky update helper (simplified: recompute every step for simplicity, N=5 is small)
        
        while calls + lam <= budget:
            # Generate population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(N)
                x = mean + sigma * (C_sqrt @ z)
                x = np.clip(x, bounds_low, bounds_high)
                offspring.append(x)
            
            # Evaluate
            fitnesses = []
            for x in offspring:
                if calls >= budget: break
                f = func(x)
                calls += 1
                fitnesses.append(f)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            if len(fitnesses) < lam:
                break
                
            # Sort by fitness
            indices = np.argsort(fitnesses)
            offspring = [offspring[i] for i in indices]
            fitnesses = [fitnesses[i] for i in indices]
            
            # Select best mu
            y = np.array([offspring[i] - mean for i in range(mu)])
            
            # Update mean
            mean = mean + sigma * (w @ y)
            mean = np.clip(mean, bounds_low, bounds_high)
            
            # Update p_sigma
            # Need eigen decomposition of C for the norm
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            p_sigma = (1.0 - cs) * p_sigma + np.sqrt(cs * (2.0 - cs) * mueff) * (C_inv_sqrt @ (y.T @ w)).T
            
            # Update step size
            # hsig condition
            # Simplified: check if p_sigma is not too large
            # Norm of p_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Expected norm of N(0, I)
            chi_N = np.sqrt(N) * (1.0 - 1.0/(4.0*N) + 1.0/(21.0*N**2))
            
            # Heuristic for hsig to avoid over-adaptation
            hsig = 1.0 if (norm_p_sigma / np.sqrt(1.0 - (1.0 - cs)**2)) < (1.4 + 2.0/(N+1)) * chi_N else 0.0
            
            sigma = sigma * np.exp(cs / (N + 2.0) * (norm_p_sigma / chi_N - 1.0))
            
            # Update p_c
            p_c = (1.0 - cc) * p_c + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * (y.T @ w).T
            
            # Update covariance matrix
            # Delta h correction
            delta_h = (1.0 - cc) * np.linalg.norm(p_c)**2 / N - 1.0
            delta_h = np.clip(delta_h, -1.0, 1.0)
            
            # Standard formula: C = (1 - c1 - cmu)C + c1 (p_c p_c^T + (1 - hsig) cc(2-cc)C) + cmu sum(w_i (y_i y_i^T))
            # Note: y is (mu, N). We need sum w_i y_i y_i^T
            # y is (mu, N). w is (mu,).
            # sum w_i y_i y_i^T = (w[:, None] * y) @ y.T
            
            sum_w_y_yT = (w[:, None] * y) @ y.T
            
            C = (1.0 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + (1.0 - hsig) * cc * (2.0 - cc) * C) + cmu * sum_w_y_yT
            
            # Symmetrize
            C = (C + C.T) / 2.0
            
            # Ensure positive definiteness (simple fix: add epsilon to diagonal if needed)
            # Recompute Cholesky for next step
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: add identity
                C = C + 1e-4 * np.eye(N)
                C_sqrt = np.linalg.cholesky(C)
                
        return best_x, best_f
