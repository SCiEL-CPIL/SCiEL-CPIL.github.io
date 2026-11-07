import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        deadline = time.time() + 290

        # CMA-ES Parameters
        mu = max(1, int(0.5 * n))
        lam = 2 * mu
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 - 1 / mu_eff + 2 * mu_eff / n) / (n + 1 + 3 * mu_eff / n)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (n + 2 + mu_eff))

        # Initial State
        x0 = np.zeros(n)
        sigma = 1.0
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        f_best = func(x0)
        x_best = x0.copy()
        calls = 1
        
        while calls + lam <= budget:
            if time.time() > deadline:
                break
                
            # Generate
            z = np.random.randn(lam, n)
            y = x0 + sigma * (z @ (D[:, np.newaxis] * B.T))
            
            # Evaluate
            f_vals = np.zeros(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                f_vals[i] = func(y[i])
                calls += 1
                if f_vals[i] < f_best:
                    f_best = f_vals[i]
                    x_best = y[i].copy()
            
            if calls >= budget:
                break

            # Sort
            order = np.argsort(f_vals)
            y = y[order]
            f_vals = f_vals[order]
            
            # Update mean
            m = np.sum(weights[:, np.newaxis] * y[:mu], axis=0)
            
            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / (sigma * D) * (B.T @ (m - x0))
            if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (n + 1)) * np.sqrt((1 - (1 - c_sigma)**(calls + 1))):
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance
            pc = (1 - c_c) * pc + c_c * (m - x0) / sigma
            C = (1 - c_1 - c_mu) * C + c_1 * (2 - c_1) * np.eye(n) + c_mu * np.outer(pc, pc)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Eigenvalue decomposition for B and D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12) # Avoid zero
            D = np.sqrt(eigvals)
            B = eigvecs
            
            x0 = m

        return x_best, f_best
