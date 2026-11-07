"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES implementation for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin of 10s

        n = self.dim
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(n))
        if lam < 2: lam = 2
        
        # Initial step size
        sigma = 2.0 
        # Initial mean: center of the box
        x = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        
        # Cumulation paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Weights
        mu = lam // 2
        w = np.log(lam + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt(mu_eff / (n + 1) - 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # State
        best_x = x.copy()
        best_f = func(x)
        
        calls_made = 1
        gen = 0
        
        while calls_made < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate offspring
            ys = np.zeros((lam, n))
            fs = np.zeros(lam)
            
            # Sample from multivariate normal using Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes singular, reset
                C = np.eye(n)
                L = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                sigma = 1.0
                
            for i in range(lam):
                if calls_made >= self.budget:
                    break
                z = np.random.randn(n)
                y = x + sigma * (L @ z)
                # Clip to box
                y = np.clip(y, -5, 5)
                ys[i] = y
                fs[i] = func(y)
                calls_made += 1
            
            if calls_made >= self.budget:
                break
            
            # If we didn't fill the whole population, adjust lam for this iteration
            # For simplicity, we assume we always fill it unless budget is hit exactly.
            # If budget is hit in the middle of a generation, we stop.
            
            # Sort offspring
            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()
                
            # Update mean
            x_new = np.zeros(n)
            for i in range(mu):
                x_new += w[i] * ys[i]
            
            # Update step size and covariance
            # y_bar is the step of the new mean
            y_bar = (x_new - x) / sigma
            
            # L_inv for sigma update
            try:
                L_inv = np.linalg.inv(L).T
            except:
                L_inv = np.eye(n)
                
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L_inv @ y_bar)
            
            # h_sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            gen += 1
            denom = np.sqrt(1 - (1 - c_sigma)**(2 * gen)) if gen > 0 else 1.0
            h_sigma = (np.linalg.norm(p_sigma) / denom) / chi_n
            h_sigma = 1 if h_sigma < 1.4 + 2 / (n + 1) else 0
            
            # sigma update
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(1e-6, min(sigma, 50.0))
            
            # p_c update
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_bar
            
            # C update
            C_new = (1 + c_1 * (h_sigma * (1 + c_c) - c_c) - c_1 - c_mu) * C
            C_new += c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                y_i = (ys[i] - x) / sigma
                C_new += c_mu * w[i] * np.outer(y_i, y_i)
                
            # Symmetrize
            C = (C_new + C_new.T) / 2
            
            # Ensure positive definiteness
            try:
                eigvals = np.linalg.eigvalsh(C)
                if eigvals[0] < 1e-6:
                    C += (1e-6 - eigvals[0]) * np.eye(n)
            except:
                C = np.eye(n)
                
            # Update mean
            x = x_new
            
            # Stop if sigma is too small
            if sigma < 1e-4:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
