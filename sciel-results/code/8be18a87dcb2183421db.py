import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Time guard
        start_time = time.time()
        deadline = start_time + 270.0

        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(n))
        # Step size
        sigma = 2.0
        # Mean
        x = np.zeros(n)
        # Covariance matrix
        C = np.eye(n)
        # Inverse square root of C (for sampling)
        # We can use Cholesky or eigen-decomp. For small n, eigen is stable.
        eigvals, eigvecs = np.linalg.eigh(C)
        B = eigvecs
        D = 1.0 / np.sqrt(np.maximum(eigvals, 1e-12))
        
        # Evolution path
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Learning rates
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        best_x = x.copy()
        best_f = float('inf')
        evals_left = budget

        while evals_left > 0 and time.time() < deadline:
            # Sample points
            ys = np.zeros((lam, n))
            xs = np.zeros((lam, n))
            fs = np.zeros(lam)
            
            for i in range(lam):
                z = np.random.randn(n)
                y = B @ (D * z)
                ys[i] = y
                xs[i] = x + sigma * y
                
                # Clip to bounds? The task says func clips. 
                # It's better to let func clip, but CMA-ES assumes continuous space.
                # The prompt says "func(x) clips... and returns a float".
                # So we can just pass xs[i].
                
                fs[i] = func(xs[i])
                evals_left -= 1
                
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = xs[i].copy()
                    
                if evals_left <= 0:
                    break

            if evals_left <= 0:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            xs = xs[order]
            fs = fs[order]
            
            # Update mean
            x_new = np.sum(weights[:, None] * xs[:mu], axis=0)
            
            # Update evolution path for sigma
            y_mean = (x_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            
            # Hessian norm check for sigma update
            hsig = np.sqrt((1 - (1 - c_sigma) ** (2 * (evals_left // lam + 1))) / (1 - (1 - c_sigma) ** 2)) if False else 1.0 # Simplified, standard CMA uses counter
            # Actually, let's use a simpler sigma update rule or standard one.
            # Standard: chi_n = sqrt(n) * (1 - 1/(4n) - 1/(21n^2))
            chi_n = np.sqrt(n) * (1 - 1.0/(4*n) - 1.0/(21*n*n))
            
            # We need to track the number of generations for the hsig condition properly.
            # For simplicity and robustness in this short budget, we can use a simplified sigma update
            # or just rely on the path. Let's use the standard path update.
            
            # The term `1 - (1 - c_sigma)^(2 * gen)` requires generation count.
            # Let's approximate hsig = 1.0 for now, or calculate it properly if we track gen.
            # Given the short budget, a simplified update is often safer to avoid bugs.
            # Let's use: sigma *= exp(c_sigma/d_sigma * (||p_sigma||/chi_n - 1))
            
            p_sigma_norm = np.linalg.norm(p_sigma)
            sigma *= np.exp(c_sigma / d_sigma * (p_sigma_norm / chi_n - 1))
            
            # Update evolution path for C
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            
            # Rank-1 and Rank-mu update
            hsig = 1.0 # Placeholder, proper calculation is complex with restarts/time guards
            # For a robust simple implementation:
            C = (1 + c_1 * hsig * hsig * c_sigma * (2 - c_sigma) - c_1 - c_mu) * C
            C += c_1 * np.outer(p_c, p_c)
            C += c_mu * np.sum(weights[:, None, None] * np.outer(ys[:mu], ys[:mu]) * np.ones((1,1,1)), axis=0)
            
            # Ensure C is positive definite (symmetrize and clip eigenvalues)
            C = (C + C.T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            x = x_new
            B = eigvecs
            D = 1.0 / np.sqrt(eigvals)

        return best_x, best_f
