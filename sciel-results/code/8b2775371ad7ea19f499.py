"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(n))
        lam = int(4 + 3 * np.log(n))
        
        # Weights for ranking
        mu = lam // 2  # Number of best individuals to update mean
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        # Initial parameters
        x_mean = np.zeros(n)
        sigma = 2.0  # Initial step size, covering [-2, 2] roughly, will adapt
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Update rates
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Damping
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        calls_made = 0
        best_f = np.inf
        best_x = None
        
        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(np.asarray(x, dtype=float).reshape(-1), -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Main Loop
        while calls_made < budget:
            # Generate population
            # We need to ensure we don't exceed budget in the generation/evaluation step
            # If remaining budget < lam, we only evaluate the remaining ones (or just break if we can't form a population, 
            # but simple truncation is safer for budget constraints)
            
            remaining = budget - calls_made
            current_lam = min(lam, remaining)
            
            if current_lam < 2:
                break
                
            # Sample
            # Cholesky factorization of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset or fix
                C = np.eye(n)
                L = np.eye(n)
                
            # Standard normal samples
            z_samples = np.random.randn(current_lam, n)
            x_samples = np.tile(x_mean, (current_lam, 1)) + sigma * (z_samples @ L.T)
            
            # Evaluate
            f_values = []
            for i in range(current_lam):
                if calls_made >= budget:
                    break
                fx = evaluate(x_samples[i])
                f_values.append((fx, i))
                
            if len(f_values) < 2:
                break
                
            # Sort by fitness
            f_values.sort(key=lambda x: x[0])
            
            # Select best mu
            mu_actual = min(mu, len(f_values))
            if mu_actual < 2:
                 break
                 
            selected_indices = [idx for _, idx in f_values[:mu_actual]]
            selected_x = x_samples[selected_indices]
            
            # Update mean
            x_mean_new = np.dot(weights[:mu_actual], selected_x)
            
            # Update covariance matrix
            # ps and pc updates
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (L.T @ (x_mean_new - x_mean))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / chi_n) < (1.4 + 2 / (n + 1))
            
            pc = (1 - cc) * pc + hsig * cc * (x_mean_new - x_mean)
            
            # Rank-one and rank-mu updates
            C = (1 + c1 * hsig * (2 - c1) - c1) * C + c1 * np.outer(pc, pc)
            
            # Rank-mu update
            y_vecs = (selected_x - x_mean) / sigma
            C += cmu * np.sum(weights[:mu_actual, None, None] * y_vecs[:, None, :] * y_vecs[:, :, None], axis=0)
            
            # Ensure symmetry and positive definiteness (small correction if needed)
            C = (C + C.T) / 2
            # Check eigenvalues
            eigvals = np.linalg.eigvalsh(C)
            if np.any(eigvals < 1e-9):
                 C += 1e-9 * np.eye(n)
                 
            x_mean = x_mean_new
            
            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Clip sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-6, 1e6)
            
            # Convergence check: if sigma is very small or variance is low
            if sigma < 1e-8:
                break

        if best_x is None:
            best_x = np.zeros(n)
            best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
