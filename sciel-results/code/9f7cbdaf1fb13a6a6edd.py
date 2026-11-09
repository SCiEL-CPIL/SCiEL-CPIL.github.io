import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s margin for safety
        n = self.dim
        total_budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Global exploration with 100 samples
        n_global = min(100, total_budget // 5)
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        for i in range(n_global):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            eval_count += 1
        
        # Phase 2: CMA-ES local refinement
        remaining = total_budget - eval_count
        if remaining > 0:
            best_x, best_f = self._cma_es(
                func, best_x, best_f, remaining, n, rng, deadline
            )
        
        return best_x, best_f
    
    def _cma_es(self, func, x0, f0, budget, dim, rng, deadline):
        # CMA-ES implementation
        n = dim
        lam = max(4 + int(3 * np.log(n)), 10)  # population size
        sigma = 1.0
        x = x0.copy()
        fx = f0
        C = np.eye(n)
        C_inv = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        cs = (1 + 2 / (n + 1)) / (n + 4)
        cmu = min(0.1, 2 / ((n + 2.2)**2 + 10))
        cmu = max(0.1, cmu)  # actually standard is 2/((n+2.2)^2+10)
        cmu = 2 / ((n + 2.2)**2 + 10)
        cc = (4 + n / 4) / (n + 4 + 2 * n / 4)
        damps = (1 + 2 * max(0, np.sqrt((cmu - 1) / (n + 1)) - 1) + cs)
        dampc = (1 + 2 * max(0, np.sqrt((cmu - 1) / (n + 1)) - 1) + cc)
        
        # Initial sigma based on best point spread - use 1.0 for simplicity
        sigma = 1.0
        
        eval_count = 0
        while eval_count < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            fvals = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                # Transform z using Cholesky of C
                y = x + sigma * (C @ z)  # approximate, should use Cholesky
                y = np.clip(y, -5, 5)
                fy = func(y)
                eval_count += 1
                ys.append(y)
                fvals.append(fy)
            
            if len(fvals) == 0:
                break
            
            # Sort by fitness
            order = np.argsort(fvals)
            ys_sorted = [ys[i] for i in order]
            fvals_sorted = [fvals[i] for i in order]
            
            # Update best
            if fvals_sorted[0] < fx:
                fx = fvals_sorted[0]
                x = ys_sorted[0].copy()
            
            # Update mean
            w = np.log(np.arange(1, lam // 2 + 1)) - np.log(lam // 2)
            w = w / np.sum(w)
            mueff = int(np.sum(w[:lam//2])**2 / np.sum(w[:lam//2]**2))
            
            m_new = sum(w[i] * ys_sorted[i] for i in range(min(lam//2, len(ys_sorted))))
            
            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv @ (m_new - x) / sigma)
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (n + 1)) * np.sqrt(n + 1 - 2) else 0
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc)) * (m_new - x) / sigma
            
            # Update covariance
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 - cs * (2 - cs) - cmu) * C + cs * np.outer(ps, ps) + cmu * sum(w[i] * np.outer(ys_sorted[i] - x, ys_sorted[i] - x) for i in range(min(lam//2, len(ys_sorted))))
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization
            C += 1e-12 * np.eye(n)
            
            # Update C_inv using Cholesky
            try:
                L = np.linalg.cholesky(C)
                C_inv = np.linalg.inv(L.T @ L)
            except np.linalg.LinAlgError:
                C_inv = np.linalg.pinv(C)
            
            x = m_new
            
            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(n + 1 - 2) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
        
        return x, fx
