"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        lb = -5.0
        ub = 5.0
        
        # CMA-ES Parameters
        lam = int(4 + 3 * np.log(n)) 
        if lam < 2: lam = 2
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        sigma = 2.0
        x_mean = np.random.uniform(lb, ub, size=n)
        
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)
        D = 1.0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        calls = 0
        best_f = float('inf')
        best_x = None
        
        def evaluate(x):
            nonlocal calls, best_f, best_x
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        f_mean = evaluate(x_mean)
        
        while calls < budget:
            ys = []
            xs = []
            fs = []
            
            for k in range(lam):
                if calls >= budget:
                    break
                z = np.random.normal(size=n)
                y = D * C_inv_sqrt.T @ z
                x = x_mean + sigma * y
                x = np.clip(x, lb, ub)
                
                fx = evaluate(x)
                ys.append(y)
                xs.append(x)
                fs.append(fx)
                
            if not fs:
                break
            
            indices = np.argsort(fs)
            ys_sorted = [ys[i] for i in indices]
            xs_sorted = [xs[i] for i in indices]
            fs_sorted = [fs[i] for i in indices]
            
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += w[i] * ys_sorted[i]
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / D * C_inv_sqrt.T @ y_w
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls - n)))) < (1.4 + 2 / (n + 1)) * chi_n
            if hsig:
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * y_w
            
            # Update Covariance Matrix
            C = (1 + c1 * (hsig * (2 - cc) - cc) - c1) * C
            if hsig:
                C += c1 * np.outer(y_w, y_w)
            
            # Rank-mu update
            for i in range(mu):
                C += cmu * w[i] * np.outer(ys_sorted[i], ys_sorted[i])
            
            # Update D and C_inv_sqrt
            C = (C + C.T) / 2
            vals, vecs = np.linalg.eigh(C)
            vals = np.clip(vals, 1e-12, None)
            D = np.sqrt(vals)
            C_inv_sqrt = vecs @ np.diag(1.0 / D) @ vecs.T
            
            # Update sigma
            sigma = sigma * np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update x_mean
            x_mean = x_mean + sigma * y_w
            
            if calls >= budget:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
