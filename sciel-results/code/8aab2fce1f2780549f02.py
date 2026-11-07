import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_x = None
        best_f = float('inf')
        
        def timed_call(x):
            nonlocal n_calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return float(val)

        rng = np.random.default_rng(42)

        # 1. Initial Global Exploration
        n_init = max(10, int(budget * 0.2))
        
        for _ in range(n_init):
            if n_calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            f = timed_call(x)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = timed_call(best_x)

        # 2. CMA-ES with Restarts
        lam = max(4, int(4 + 3 * math.log(dim)))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
        sigma_path = 1 + 2 * max(0, (cs - 1) / (dim + 1)) - cs
        
        stagnation_threshold = 1e-6
        stagnation_generations = 10
        calls_since_improve = 0
        best_f_last_restart = best_f
        
        cmaes_active = False
        cmaes_mean = None
        cmaes_sigma = None
        cmaes_C = None
        cmaes_pc = None
        cmaes_ps = None

        while n_calls < budget and time.time() < deadline:
            if budget - n_calls < lam:
                while n_calls < budget and time.time() < deadline:
                    step_size = max(0.01, 0.1 * (1 + (best_f_last_restart - best_f) / (1 + abs(best_f_last_restart))))
                    y = np.clip(best_x + step_size * rng.standard_normal(dim), -5.0, 5.0)
                    f_y = timed_call(y)
                    if f_y < best_f:
                        best_f = f_y
                        best_x = y.copy()
                        calls_since_improve = 0
                    else:
                        calls_since_improve += 1
                break

            if not cmaes_active:
                cmaes_mean = best_x.copy()
                cmaes_sigma = 0.5
                cmaes_C = np.eye(dim)
                cmaes_pc = np.zeros(dim)
                cmaes_ps = np.zeros(dim)
                cmaes_active = True
            else:
                cmaes_mean = cmaes_mean
                cmaes_sigma = cmaes_sigma
                cmaes_C = cmaes_C
                cmaes_pc = cmaes_pc
                cmaes_ps = cmaes_ps

            Y = []
            FY = []
            for i in range(lam):
                if n_calls >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                try:
                    L = np.linalg.cholesky(cmaes_C + 1e-10 * np.eye(dim))
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                
                y = np.clip(cmaes_mean + cmaes_sigma * (L @ z), -5.0, 5.0)
                fy = timed_call(y)
                Y.append(y)
                FY.append(fy)
            
            if len(Y) < lam:
                break

            idx = np.argsort(FY)
            Y_sorted = [Y[i] for i in idx]
            FY_sorted = [FY[i] for i in idx]
            
            if FY_sorted[0] < best_f:
                best_f = FY_sorted[0]
                best_x = Y_sorted[0].copy()
                calls_since_improve = 0
            else:
                calls_since_improve += lam
            
            new_mean = np.sum([weights[i] * Y_sorted[i] for i in range(mu)], axis=0)
            
            try:
                eigvals, eigvecs = np.linalg.eigh(cmaes_C)
                eigvals_clipped = np.clip(eigvals, 1e-12, None)
                C_inv_sqrt = eigvecs @ np.diag(1/np.sqrt(eigvals_clipped)) @ eigvecs.T
                cmaes_ps = (1 - cs) * cmaes_ps + cs / np.sqrt(mu_eff) * (C_inv_sqrt @ ((new_mean - cmaes_mean) / cmaes_sigma))
            except:
                cmaes_ps = (1 - cs) * cmaes_ps + cs / np.sqrt(mu_eff) * ((new_mean - cmaes_mean) / cmaes_sigma)

            cmaes_sigma *= np.exp(cs / sigma_path * (np.linalg.norm(cmaes_ps) / np.sqrt(2 * dim) - 1))
            
            cmaes_pc = (1 - c1) * cmaes_pc + c1 / np.sqrt(mu_eff) * (new_mean - cmaes_mean) / cmaes_sigma
            
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_diff = (Y_sorted[i] - cmaes_mean) / cmaes_sigma
                rank_mu += weights[i] * np.outer(y_diff, y_diff)
            
            cmaes_C = (1 - c1 - cmu) * cmaes_C + c1 * (np.outer(cmaes_pc, cmaes_pc) + (1 + (1 - cs) * (2 - cs)) * cs * (1 - cs) * np.outer(cmaes_ps, cmaes_ps)) + cmu * rank_mu
            
            cmaes_C = (cmaes_C + cmaes_C.T) / 2
            if not np.all(np.isfinite(cmaes_C)):
                cmaes_C = np.eye(dim)
            
            cmaes_mean = new_mean

            if calls_since_improve >= stagnation_generations * lam:
                if best_f > best_f_last_restart - stagnation_threshold:
                    best_f_last_restart = best_f
                    calls_since_improve = 0
                    cmaes_active = False
                    x_new = rng.uniform(-5.0, 5.0, dim)
                    f_new = timed_call(x_new) if n_calls < budget else float('inf')
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                    continue

        return best_x, best_f
