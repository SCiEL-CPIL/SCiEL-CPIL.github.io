"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Population size for CMA-ES
        lam = 4 + int(3 * np.log(n))
        # Ensure we don't exceed budget
        if lam > budget - 10:
            lam = max(2, budget - 10)
        
        # Initial point: center of the box
        x_mean = np.zeros(n)
        # Initial sigma: large enough to explore
        sigma = 2.0
        
        # CMA-ES parameters
        mu = lam // 2  # number of selected parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c1 = (2 + 2.0 / (n + 1.3)) / ((n + 1) ** 2 + 2)
        cmu = min(1 - c1, (2 + (mu_eff - 2) / (n + 1)) / ((n + 2) ** 2 + mu_eff))
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        ds = 1 + cs + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1)
        cc = (4 + n / 4) / (n + 4 + 2 * mu_eff / n)
        
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        C_inv_sqrt = np.eye(n)
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        call_count = 0
        best_x = x_mean.copy()
        best_f = np.inf
        
        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            call_count += 1
            fx = func(np.clip(x, -5, 5))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Initial evaluation
        f_x = wrapped_func(x_mean)
        
        # CMA-ES main loop
        while call_count < budget:
            # Generate population
            if call_count + lam > budget:
                # Only evaluate remaining budget
                remaining = budget - call_count
                if remaining <= 0:
                    break
                # Use partial population
                indices = np.random.permutation(lam)[:remaining]
                pops = []
                fvals = []
                for i in indices:
                    z = np.random.randn(n)
                    y = x_mean + sigma * (C_sqrt @ z)
                    fy = wrapped_func(y)
                    pops.append(y)
                    fvals.append(fy)
                if len(pops) < 2:
                    break
                # Sort by fitness
                order = np.argsort(fvals)
                # Select best mu
                mu_actual = min(mu, len(pops))
                y_selected = [pops[i] for i in order[:mu_actual]]
                w_selected = weights[:mu_actual] / np.sum(weights[:mu_actual])
                
                # Update mean
                x_new = np.sum([w * y for w, y in zip(w_selected, y_selected)], axis=0)
                
                # Update covariance (simplified)
                diff = [y - x_mean for y in y_selected]
                diff = np.array(diff)
                C_new = C + cmu / mu_eff * np.sum([w * np.outer(d, d) for w, d in zip(w_selected, diff)], axis=0)
                C_new = (C_new + C_new.T) / 2
                C = C_new
                C_inv = np.linalg.inv(C + 1e-8 * np.eye(n))
                C_sqrt = np.linalg.cholesky(C + 1e-8 * np.eye(n))
                
                x_mean = x_new
                sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                break  # We used all remaining budget
            
            # Full population generation
            pops = []
            fvals = []
            for i in range(lam):
                z = np.random.randn(n)
                y = x_mean + sigma * (C_sqrt @ z)
                fy = wrapped_func(y)
                pops.append(y)
                fvals.append(fy)
            
            # Sort by fitness
            order = np.argsort(fvals)
            y_sorted = [pops[i] for i in order]
            f_sorted = [fvals[i] for i in order]
            
            # Select best mu
            y_selected = y_sorted[:mu]
            
            # Update mean
            x_new = np.sum([w * y for w, y in zip(weights, y_selected)], axis=0)
            
            # Update evolution paths
            y_diff = np.array([y - x_mean for y in y_selected])
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (C_sqrt @ (weights @ (C_inv_sqrt @ y_diff.T).T))
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (call_count + lam) / (n + 1)))) / chi_n
            h_sigma = min(1.0, h_sigma)
            
            p_c = (1 - cs) * p_c + h_sigma * np.sqrt(cs * (2 - cs) * mu_eff) * (C_inv_sqrt @ (weights @ y_diff.T))
            
            # Update covariance
            diff = np.array([y - x_mean for y in y_selected])
            C = (1 + c1 + cmu - c_mu) * C + c1 * np.outer(p_c, p_c) + cmu * np.sum([w * np.outer(d, d) for w, d in zip(weights, diff)], axis=0)
            C = (C + C.T) / 2
            
            # Ensure positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            C_inv = np.linalg.inv(C + 1e-8 * np.eye(n))
            C_sqrt = np.linalg.cholesky(C + 1e-8 * np.eye(n))
            C_inv_sqrt = np.linalg.cholesky(C_inv + 1e-8 * np.eye(n))
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            x_mean = x_new
        
        return best_x, best_f
# EVOLVE-BLOCK-END
