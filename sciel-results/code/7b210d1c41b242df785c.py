"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initialize parameters
        x0 = rng.uniform(-3, 3, n)
        sigma0 = 1.5
        
        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(n)))  # population size
        c_sigma = (mu_eff + 2) / ((n + 1.3) ** 2 + mu_eff)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / ((n + 1.3) ** 2 + mu_eff)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # State variables
        x = x0.copy()
        sigma = sigma0
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        best_x = x0.copy()
        best_f = func(x0)
        calls = 1
        
        # Generation loop
        while calls < budget:
            # Generate offspring
            ys = np.zeros((lam, n))
            fs = np.zeros(lam)
            
            # Adaptive step size based on recent performance
            # Simple adaptation: use sigma directly
            
            for i in range(lam):
                z = rng.standard_normal(n)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                ys[i] = y
                fs[i] = func(y)
                calls += 1
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
                
            # Sort by fitness
            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()
                
            # Compute weighted mean of top mu
            y_w = np.sum(weights[:, np.newaxis] * ys[:mu], axis=0)
            
            # Update evolution path
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (B @ (D * (y_w - x)))
            hsig = (np.sqrt(1 - (1 - c_c) ** (2 * (calls // lam))) * chi_n) / (1 - c_c)
            ps = (1 - c_sigma) * ps + np.hypot(np.dot(B.T, pc) / (D * sigma)) < (1.4 + 2 / (n + 1)) * hsig * np.sqrt(c_sigma * (2 - c_sigma)) * np.sign(np.dot(B.T, pc) / (D * sigma))
            
            # Actually, let's simplify the ps update
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * (y_w - x)))
            
            # Update covariance matrix
            delta_hsig = (1 - np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)) * np.linalg.norm(ps) / chi_n - 1
            C_update = c1 * (np.outer(pc, pc) - np.eye(n)) + cmu * np.sum(
                weights[:, np.newaxis, np.newaxis] * np.outer(
                    B @ (D * (ys[:mu] - x) / sigma), 
                    B @ (D * (ys[:mu] - x) / sigma)
                ), axis=0
            ) + c1 * delta_hsig * np.eye(n)
            
            # Update B and D using eigenvalue decomposition (approximate for small n)
            # For efficiency, we can use a simplified update
            eigval, eigvec = np.linalg.eigh(np.eye(n) + C_update * sigma ** 2)
            eigval = np.maximum(eigval, 1e-12)
            B = eigvec
            D = np.sqrt(eigval)
            
            # Update mean
            x = y_w
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Ensure x stays in bounds
            x = np.clip(x, -5, 5)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
