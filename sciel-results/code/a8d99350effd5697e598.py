import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        deadline = time.time() + 280.0
        dim = self.dim
        budget = self.budget
        calls = 0
        
        # Phase 1: Brief random exploration for early progress
        n_init = 5
        best_x = None
        best_f = np.inf
        for _ in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            best_f = func(best_x)
            calls += 1
        
        # Phase 2: CMA-ES
        x0 = best_x.copy()
        sigma0 = 1.5
        result = self._cma_es(func, x0, sigma0, budget - calls, dim, rng, deadline, calls)
        # result is (best_x, best_f)
        return result[0], result[1]
    
    def _cma_es(self, func, x0, sigma0, remaining_budget, dim, rng, deadline, calls_ref):
        n = dim
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2
        weights = np.log(mu + 1) - np.arange(1, mu + 1)
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        c1 = (2 - mueff / n + cs) / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        
        x = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        best_x = x0.copy()
        best_f = func(x0)
        calls_ref += 1
        
        no_improve = 0
        max_no_improve = 30  # generations without improvement
        
        while calls_ref < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            for i in range(lam):
                if calls_ref >= budget:
                    break
                if time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                y = np.clip(x + sigma * C_inv_sqrt @ z, -5, 5)
                fy = func(y)
                calls_ref += 1
                ys.append((fy, y))
            
            if len(ys) < 2:
                break
            
            ys.sort(key=lambda t: t[0])
            
            if ys[0][0] < best_f:
                best_f = ys[0][0]
                best_x = ys[0][1].copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Mean of selected
            y_mean = sum(weights[i] * ys[i][1] for i in range(mu))
            
            d = (y_mean - x) / sigma
            
            # Evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ d
            ps_norm = np.linalg.norm(ps)
            # hsig threshold
            hsig_thresh = (1.4 + 2 / (n + 1)) * np.sqrt(1 - (1 - cs)**2) * np.sqrt(mueff)
            hsig = ps_norm < hsig_thresh
            pc = (1 - cc) * pc + hsig * cc * d
            
            # Covariance update
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(ps, ps) + (hsig * (2 / cc - 1) - 1) * np.eye(n))
            for i in range(mu):
                diff = ys[i][1] - x
                C += cmu * weights[i] * np.outer(diff, diff)
            
            # Re-orthogonalize C via eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Update mean and sigma
            x = y_mean
            sigma *= np.exp(cs / damps * (ps_norm / np.sqrt(mueff) - 1))
            sigma = max(sigma, 1e-10)
            
            # Stagnation check
            if no_improve > max_no_improve or sigma < 1e-6:
                # Restart: perturb best point
                if calls_ref >= budget - lam:
                    break
                x = np.clip(best_x + 0.5 * rng.standard_normal(n), -5, 5)
                fx = func(x)
                calls_ref += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma = 1.0
                C = np.eye(n)
                C_inv_sqrt = np.eye(n)
                ps = np.zeros(n)
                pc = np.zeros(n)
                no_improve = 0
        
        return best_x, best_f
