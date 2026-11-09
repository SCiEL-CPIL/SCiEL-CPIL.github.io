"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with 1/5 success rule and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng):
        n = self.dim
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        xmean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        D = np.linalg.cholesky(C).diagonal()
        inv_D = 1.0 / D
        B = np.linalg.inv(np.diag(D))
        
        best_x = x0.copy()
        best_f = func(x0)
        calls_used = 1
        
        for gen in range(budget // lam + 1):
            if calls_used >= budget:
                break
            if time.time() - self.start_time > self.deadline:
                break
                
            ys = []
            fs = []
            for i in range(lam):
                z = rng.standard_normal(n)
                y = np.clip(xmean + sigma * (B @ z), -5, 5)
                fy = func(y)
                ys.append(y)
                fs.append(fy)
                calls_used += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            fs = [fs[i] for i in order]
            
            ymean = np.sum(w[:, None] * np.array(ys[:mu]), axis=0)
            v = (ymean - xmean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * v
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * v
            
            if hsig:
                C = (1 + c1 + cmu - c1) * C - c1 * np.outer(pc, pc) + cmu * np.sum(w[:mu, None, None] * np.outer(np.array(ys[:mu]) - xmean, np.array(ys[:mu]) - xmean) / sigma**2, axis=0)
            
            # Step size adaptation (1/5 rule)
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            xmean = ymean
            
            # Update Cholesky
            try:
                D = np.linalg.cholesky(C + 1e-20 * np.eye(n))
                inv_D = 1.0 / D
                B = np.linalg.inv(np.diag(D))
            except np.linalg.LinAlgError:
                C = np.eye(n)
                D = np.ones(n)
                inv_D = np.ones(n)
                B = np.eye(n)
                
        return best_x, best_f, calls_used

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = 250.0  # Safety margin for 300s limit
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = float('inf')
        remaining = self.budget
        restart_count = 0
        
        while remaining > 0 and time.time() - self.start_time < self.deadline:
            # Initial sigma based on box size
            sigma0 = 1.0
            
            # Random restart location, biased towards center slightly
            x0 = rng.uniform(-4, 4, self.dim)
            
            x, f, used = self._cma_es(func, x0, sigma0, remaining, rng)
            
            if f < best_f:
                best_f = f
                best_x = x
            
            remaining -= used
            restart_count += 1
            
            # If we made very little progress or used very few calls, restart
            if used < 10:
                break
                
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
